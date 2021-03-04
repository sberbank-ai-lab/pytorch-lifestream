from make_datasets_spark import DatasetConverter

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType, FloatType, ArrayType
from pyspark.sql.window import Window
from pyspark.sql.functions import dense_rank
import os
import json
import logging
import datetime
import numpy as np


FILE_NAME_TRAIN = 'train.csv'
FILE_NAME_TEST = 'test.csv'


logger = logging.getLogger(__name__)


class LocalDatasetConverter(DatasetConverter):
    def spark_read_file(self, path):
        spark = SparkSession.builder.getOrCreate()
        # merge to base class 
        ext = os.path.splitext(path)[1]
        if ext == '.csv':
            return spark.read.option("escape", "\"").csv(path, header=True).limit(10**6)
        else:
            __super().spark_read_file(path)

    def load_transactions(self):
        df_train = self.spark_read_file(self.path_to_file(FILE_NAME_TRAIN))
        df_test = self.spark_read_file(self.path_to_file(FILE_NAME_TEST))
        logger.info(f'Loaded {df_train.count()} records from "{FILE_NAME_TRAIN}"')
        logger.info(f'Loaded {df_test.count()} records from "{FILE_NAME_TEST}"')

        # TODO
        # convertation to timestamp
        # json parsing

        for col in df_train.columns:
            if col not in df_test.columns:
                df_test = df_test.withColumn(col, F.lit(None))
                logger.info(f'Test extended with "{col}" column')

        df = df_train.union(df_test)

        return df

    def trx_to_features(self, df_data, print_dataset_info,
                        col_client_id, cols_event_time, cols_category, cols_log_norm, max_trx_count):
        if print_dataset_info:
            unique_clients = df_data.select(col_client_id).distinct().count()
            logger.info(f'Found {unique_clients} unique clients')

        # Converting timestamp to float
        frmt = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        df_data = df_data.withColumn('event_time', F.unix_timestamp(cols_event_time[0], frmt))
        df_data = df_data.withColumn('event_time', F.col('event_time') / (24 * 60 * 60))

        # Process 'correct' key in json data
        udf_function = udf(lambda x: str(json.loads(x).get('correct', 'None')), StringType())
        df_data = df_data.withColumn('correct', udf_function('event_data'))

        # Label encoding
        encoders = {col: self.get_encoder(df_data, col) for col in cols_category}
        for col in cols_category:
            df_data = self.encode_col(df_data, col, encoders[col])
            if print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{self.pd_hist(df_data, col)}')
        
        for col in cols_log_norm:
            df_data = self.log_transform(df_data, col)
            if print_dataset_info:
                logger.info(f'Encoder stat for "{col}":\ncodes | trx_count\n{self.pd_hist(df_data, col)}')

        # Enumerating 'game_session' column
        window = Window.partitionBy(df_data[col_client_id]).orderBy(df_data['game_session'])
        df_data = df_data.withColumn('game_session', dense_rank().over(window))

        if print_dataset_info:
            df_temp = df_data.groupby(col_client_id).agg(F.count(F.lit(1)).alias("trx_count"))
            logger.info(f'Trx count per clients:\nlen(trx_list) | client_count\n{self.pd_hist(df_temp, "trx_count")}')

        # Column filter
        used_columns = cols_category + cols_log_norm + ['game_time', 'game_session', 'event_time', col_client_id]
        used_columns = [col for col in df_data.columns if col in used_columns]

        logger.info('Feature collection in progress ...')
        features = df_data.select(used_columns)
        # features = self.remove_long_trx(features, max_trx_count, col_client_id)
        features = self.collect_lists(features, col_client_id)

        features = features.withColumn('game_time', features.game_time.cast('array<int>'))

        if print_dataset_info:
            feature_names = list(features.columns)
            logger.info(f'Feature names: {feature_names}')

        features.persist()
        return features

    def update_with_target(self, source_df, features_df, target_df):
        data = (
            source_df.join(
                target_df.select(['game_session', 'accuracy_group']),
                on='game_session',
                how='right'
            )
            .where((F.col('event_type') == 'Assessment') & (F.col('event_code') == 2000))
            .select(['installation_id', 'timestamp', 'accuracy_group'])
        )

        frmt = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        data = data.withColumn('timestamp', F.unix_timestamp('timestamp', frmt))
        data = data.withColumn('timestamp', F.col('timestamp') / (24 * 60 * 60))
        data = data.withColumnRenamed('accuracy_group', 'target')

        data = data.join(features_df, on='installation_id', how='left')

        def get_index(event_time, timestamp):
            return int(np.searchsorted(np.array(event_time), timestamp)) + 1
            
        udf_function = udf(get_index, IntegerType())
        data = data.withColumn('index', udf_function('event_time', 'timestamp'))

        # Slice transactions
        cols_to_slice = [
            'game_session', 'game_time',
            'event_id', 'event_code', 'event_type', 
            'title', 'world', 'correct'
        ]
        max_trx_count = self.config.max_trx_count or 1200
        for col in cols_to_slice:
            udf_function = udf(lambda seq, index: seq[max(0, index - max_trx_count): index], ArrayType(IntegerType()))
            data = data.withColumn(col, udf_function(col, 'index'))
    
        udf_function = udf(lambda seq, index: seq[max(0, index - max_trx_count): index], ArrayType(FloatType()))
        data = data.withColumn('event_time', udf_function('event_time', 'index'))
        
        udf_function = udf(lambda seq: len(seq), IntegerType())
        data = data.withColumn('trx_count', udf_function('event_time'))
        data = data.drop('index')
        data = data.drop('timestamp')

        data.persist()

        return data

    def save_features(self, df_data, save_path):
        df_data.write.parquet(save_path, mode='overwrite')
        logger.info(f'Saved partitions to: "{save_path}"')

    def run(self):
        _start = datetime.datetime.now()
        self.parse_args()
        spark = SparkSession.builder.getOrCreate()

        self.logging_config()

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'load_source_data')
        source_data = self.load_transactions()

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'trx_to_features')
        client_features = self.trx_to_features(
            df_data=source_data,
            print_dataset_info=self.config.print_dataset_info,
            col_client_id=self.config.col_client_id,
            cols_event_time=self.config.cols_event_time,
            cols_category=self.config.cols_category,
            cols_log_norm=self.config.cols_log_norm,
            max_trx_count=self.config.max_trx_count,
        )

        if len(self.config.target_files) > 0 and len(self.config.col_target) > 0:
            # load target
            df_target = self.load_target()
            df_target.persist()

            if len(self.config.col_target) == 1:
                col_target = self.config.col_target[0]
            else:
                col_target = self.config.col_target

            # description
            spark.sparkContext.setLocalProperty('callSite.short', 'update_with_target')
            client_features = self.update_with_target(
                source_df=source_data,
                features_df=client_features,
                target_df=df_target,
            )

        train, test, save_test_id = None, None, False
        if self.config.test_size == 'predefined':
            train, test = self.split_dataset_predefined(
                all_data=client_features,
                data_path=self.config.data_path,
                col_client_id=self.config.col_client_id,
                test_ids_path=self.config.output_test_ids_path,
            )
        elif float(self.config.test_size) > 0:
            # description
            spark.sparkContext.setLocalProperty('callSite.short', 'split_dataset')
            train, test = self.split_dataset(
                all_data=client_features,
                test_size=float(self.config.test_size),
                df_target=df_target,
                col_client_id=self.config.col_client_id,
                salt=self.config.salt,
            )
            save_test_id = True
        else:
            train = client_features

        # description
        spark.sparkContext.setLocalProperty('callSite.short', 'save_features')
        self.save_features(
            df_data=train,
            save_path=self.config.output_train_path,
        )

        if test is not None:
            self.save_features(
                df_data=test,
                save_path=self.config.output_test_path,
            )

        if save_test_id:
            test_ids = test.select(self.config.col_client_id).distinct().toPandas()
            test_ids.to_csv(self.config.output_test_ids_path, index=False)

        _duration = datetime.datetime.now() - _start
        logger.info(f'Data collected in {_duration.seconds} sec ({_duration})')

if __name__ == '__main__':
    LocalDatasetConverter().run()

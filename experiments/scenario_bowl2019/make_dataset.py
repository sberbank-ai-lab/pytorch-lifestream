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
FILE_NAME_LABEL = 'train_labels.csv'
COL_EVENT_TIME = 'timestamp'


logger = logging.getLogger(__name__)


class LocalDatasetConverter(DatasetConverter):
    def load_transactions(self):
        df_train = self.spark_read_file(self.path_to_file(FILE_NAME_TRAIN))
        df_test = self.spark_read_file(self.path_to_file(FILE_NAME_TEST))
        logger.info(f'Loaded {df_train.count()} records from "{FILE_NAME_TRAIN}"')
        logger.info(f'Loaded {df_test.count()} records from "{FILE_NAME_TEST}"')

        for col in df_train.columns:
            if col not in df_test.columns:
                df_test = df_test.withColumn(col, F.lit(None))
                logger.info(f'Test extended with "{col}" column')

        df = df_train.union(df_test)

        # event_time mapping
        frmt = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        df = df.withColumn('event_time', F.unix_timestamp(COL_EVENT_TIME, frmt))
        df = df.withColumn('event_time', F.col('event_time') / (24 * 60 * 60))
        # df = df.drop('timestamp')

        # Process 'correct' key in json data
        udf_function = udf(lambda x: str(json.loads(x).get('correct', 'None')), StringType())
        df = df.withColumn('correct', udf_function('event_data'))
        # df = df.drop('event_data')

        return df

    def load_target(self):
        df_target = self.spark_read_file(self.path_to_file(FILE_NAME_LABEL))
        return df_target

if __name__ == '__main__':
    LocalDatasetConverter().run()

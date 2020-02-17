# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Covertype training pipeline DSL."""

import os
from typing import Dict, List, Text

from kfp import gcp
from tfx.components.base import executor_spec
from tfx.components.common_nodes.importer_node import ImporterNode
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.big_query_example_gen.component import BigQueryExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.proto import evaluator_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input
from tfx.types.standard_artifacts import Schema
from use_mysql_secret import use_mysql_secret
from modules.custom_components import AccuracyModelValidator


def _create__pipeline(pipeline_name: Text, pipeline_root: Text, dataset_name: Text,
                      ai_platform_training_args: Dict[Text, Text],
                      ai_platform_serving_args: Dict[Text, Text],
                      beam_pipeline_args: List[Text]) -> pipeline.Pipeline:
    """Implements the online news pipeline with TFX."""

    # Dataset, table and/or 'where conditions' can be passed as pipeline args.
    query='SELECT * FROM {}.census'.format(_dataset_name) 
    
    # Brings data into the pipeline from BigQuery.
    example_gen = BigQueryExampleGen(
        query=query
    )

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

    # Import schema from local directory.
    schema_importer = ImporterNode(
        instance_name='RawSchemaImporter',
        source_uri='raw_schema',
        artifact_type=Schema,
    )

    # Performs anomaly detection based on statistics and data schema.
    validate_stats = ExampleValidator(
        stats=statistics_gen.outputs.output, 
        schema=schema_importer.outputs.result
    )

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        input_data=example_gen.outputs.examples,
        schema=schema_importer.outputs.result,
        module_file='modules/transform.py'
    )

    # Train and export serving and evaluation saved models.
    trainer = Trainer(
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            ai_platform_trainer_executor.Executor),
        module_file='modules/train.py',
        transformed_examples=transform.outputs.transformed_examples,
        schema=schema_importer.outputs.result,
        transform_output=transform.outputs.transform_output,
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000),
        custom_config={'ai_platform_training_args': ai_platform_training_args}
    )

    # Uses TFMA to compute a evaluation statistics over features of a model.
    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model_exports=trainer.outputs.output,
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(
            specs=[evaluator_pb2.SingleSlicingSpec(
                column_for_slicing=['occupation'])]
        )
    )

    # Use a custom AccuracyModelValidator component to validate the model.
    model_validator = AccuracyModelValidator(
        eval_results=model_analyzer.outputs.output,
        model=trainer.outputs .model,
        accuracy_threshold=0.75,
        slice_accuracy_tolerance=0.15,
        instance_name="Accuracy_Model_Validator"
    )

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            ai_platform_pusher_executor.Executor),
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        custom_config={'ai_platform_serving_args': ai_platform_serving_args}
    )
    
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, schema_importer, validate_stats, transform,
            trainer, model_analyzer, model_validator, pusher
      ],
      # enable_cache=True,
      beam_pipeline_args=beam_pipeline_args)


if __name__ == '__main__':

    # Get settings from environment variables
    _pipeline_name = os.environ.get('PIPELINE_NAME')
    _project_id = os.environ.get('PROJECT_ID')
    _gcp_region = os.environ.get('GCP_REGION')
    _pipeline_image = os.environ.get('TFX_IMAGE')
    _dataset_name = os.environ.get('DATASET_NAME')
    _artifact_store_uri = os.environ.get('ARTIFACT_STORE_URI')
    _runtime_version = os.environ.get('RUNTIME_VERSION')
    _python_version = os.environ.get('PYTHON_VERSION')

    # AI Platform Training settings
    _ai_platform_training_args = {
        'project': _project_id,
        'region': _gcp_region,
        'masterConfig': {
            'imageUri': _pipeline_image
        }
    }

    # AI Platform Prediction settings
    _ai_platform_serving_args = {
        'model_name': 'model_' + _pipeline_name,
        'project_id': _project_id,
        'region': _gcp_region,
        'runtimeVersion': _runtime_version,
        'pythonVersion': _python_version
    }

    # Dataflow settings.
    _beam_tmp_folder = '{}/beam/tmp'.format(_artifact_store_uri)
    _beam_pipeline_args = [
        '--runner=DataflowRunner',
        '--experiments=shuffle_mode=auto',
        '--project=' + _project_id,
        '--temp_location=' + _beam_tmp_folder,
        '--region=' + _gcp_region,
    ]

    # ML Metadata settings
    _metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
    _metadata_config.mysql_db_service_host.environment_variable = 'MYSQL_SERVICE_HOST'
    _metadata_config.mysql_db_service_port.environment_variable = 'MYSQL_SERVICE_PORT'
    _metadata_config.mysql_db_name.value = 'metadb'
    _metadata_config.mysql_db_user.environment_variable = 'MYSQL_USERNAME'
    _metadata_config.mysql_db_password.environment_variable = 'MYSQL_PASSWORD'

    operator_funcs = [
        gcp.use_gcp_secret('user-gcp-sa'),
        use_mysql_secret('mysql-credential')
    ]

    # Compile the pipeline
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=_metadata_config,
        pipeline_operator_funcs=operator_funcs,
        tfx_image=_pipeline_image
    )

    _pipeline_root = '{}/{}'.format(_artifact_store_uri, _pipeline_name)
    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
        _create__pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            dataset_name=_dataset_name,
            ai_platform_training_args=_ai_platform_training_args,
            ai_platform_serving_args=_ai_platform_serving_args,
            beam_pipeline_args=_beam_pipeline_args)
    )
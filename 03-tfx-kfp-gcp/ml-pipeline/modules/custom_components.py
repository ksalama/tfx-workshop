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
"""TFX Custom Components."""

import os
import tensorflow_model_analysis as tfma
import tfx
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.components.base import base_executor
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import artifact_utils
from tfx.utils import io_utils
from typing import Optional
from tfx import types


class AccuracyValidatorSpec(tfx.types.ComponentSpec):
    
    INPUTS = {
        'eval_results': ChannelParameter(type=standard_artifacts.ModelEvaluation),
        'model': ChannelParameter(type=standard_artifacts.Model),
    }
    
    OUTPUTS = {
      'blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
    }
    
    PARAMETERS = {
        'accuracy_threshold': ExecutionParameter(type=float),
        'slice_accuracy_tolerance': ExecutionParameter(type=float),
    }

    
class AccuracyValidatorExecutor(base_executor.BaseExecutor):
    
    def Do(self, input_dict, output_dict, exec_properties):
        
        valid = True
        
        self._log_startup(input_dict, output_dict, exec_properties)
        
        accuracy_threshold = exec_properties['accuracy_threshold']
        slice_accuracy_tolerance = exec_properties['slice_accuracy_tolerance']
        min_slice_accuracy = accuracy_threshold - slice_accuracy_tolerance
        
        results_uri = input_dict['eval_results'][0].uri
        eval_results = tfma.load_eval_result(results_uri)
        
        overall_acc = eval_results.slicing_metrics[0][1]['']['']['accuracy']['doubleValue']

        if overall_acc >= accuracy_threshold:
            for slicing_metric in eval_results.slicing_metrics:
                slice_acc = slicing_metric[1]['']['']['accuracy']['doubleValue']
                if slice_acc < min_slice_accuracy:
                    valid = False
                    break
        else:
            valid = False
        
        blessing = artifact_utils.get_single_instance(output_dict['blessing'])
        
        # Current model.
        current_model = artifact_utils.get_single_instance(input_dict['model'])
        blessing.set_string_custom_property('current_model', current_model.uri)
        blessing.set_int_custom_property('current_model_id', current_model.id)

        if valid:
            io_utils.write_string_file(os.path.join(blessing.uri, 'BLESSED'), '')
            blessing.set_int_custom_property('blessed', 1)
        else:
            io_utils.write_string_file(os.path.join(blessing.uri, 'NOT_BLESSED'), '')
            blessing.set_int_custom_property('blessed', 0)


class AccuracyModelValidator(base_component.BaseComponent):

    SPEC_CLASS = AccuracyValidatorSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(AccuracyValidatorExecutor)
    
    def __init__(self,
                 eval_results: types.channel,
                 model: types.channel,
                 accuracy_threshold: float,
                 slice_accuracy_tolerance: float,
                 blessing: Optional[types.Channel] = None,
                 instance_name=None):
        
        blessing = blessing or types.Channel(
            type=standard_artifacts.ModelBlessing,
            artifacts=[standard_artifacts.ModelBlessing()])
        
        spec = AccuracyValidatorSpec(
            eval_results=eval_results, model=model, blessing=blessing, 
            accuracy_threshold=accuracy_threshold, slice_accuracy_tolerance=slice_accuracy_tolerance
        )
        
        super().__init__(spec=spec, instance_name=instance_name)
    
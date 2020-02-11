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
"""A utility function to populate environment variables from the k8s secret."""


def use_mysql_secret(secret_name='mysql-credential',
                     db_username='username',
                     db_password='password'):
  """An operator that configures the container to inject env variables with mysql credentials."""

  def _use_mysql_secret(task):
    from kubernetes import client as k8s_client
    return (task.add_env_variable(
        k8s_client.V1EnvVar(
            name='MYSQL_USERNAME',
            value_from=k8s_client.V1EnvVarSource(
                secret_key_ref=k8s_client.V1SecretKeySelector(
                    name=secret_name, key=db_username)))).add_env_variable(
                        k8s_client.V1EnvVar(
                            name='MYSQL_PASSWORD',
                            value_from=k8s_client.V1EnvVarSource(
                                secret_key_ref=k8s_client.V1SecretKeySelector(
                                    name=secret_name, key=db_password)))))

  return _use_mysql_secret
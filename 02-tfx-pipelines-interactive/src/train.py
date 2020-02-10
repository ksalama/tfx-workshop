
import math

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
from tensorflow_transform.tf_metadata import schema_utils


TARGET_FEATURE_NAME = 'income_bracket'
WEIGHT_FEATURE_NAME = 'fnlwgt'
TARGET_FEATURE_LABELS = [' <=50K', ' >50K']

# Input function for train and eval transformed data
def make_input_fn(tfrecords_files, transform_output,
                  batch_size, num_epochs=1, shuffle=False):
    
    def _gzip_reader_fn(filenames):
        return tf.data.TFRecordDataset(
            filenames, compression_type='GZIP')
    
    def input_fn():
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=tfrecords_files,
            batch_size=batch_size,
            features=transform_output.transformed_feature_spec(),
            label_key=TARGET_FEATURE_NAME,
            reader=_gzip_reader_fn,
            num_epochs=num_epochs,
            shuffle=shuffle
        )
        return dataset
    return input_fn


# Create estimator
def create_estimator(params, transform_output, run_config):
    
    feature_columns = []
    
    # Create feature columns based on the transform schema
    transformed_features = transform_output.transformed_metadata.schema.feature
    for feature in transformed_features:
        
        if feature.name in [TARGET_FEATURE_NAME, WEIGHT_FEATURE_NAME]:
            continue

        if hasattr(feature, 'int_domain') and feature.int_domain.is_categorical:
            vocab_size = feature.int_domain.max + 1
            feature_columns.append(
                tf.feature_column.embedding_column(
                    tf.feature_column.categorical_column_with_identity(
                        feature.name, num_buckets=vocab_size),
                    dimension = int(math.sqrt(vocab_size))))
        else:
            feature_columns.append(
                tf.feature_column.numeric_column(feature.name))
            
    # Create DNNClassifier        
    estimator = tf.estimator.DNNClassifier(
        weight_column=WEIGHT_FEATURE_NAME,
        label_vocabulary=TARGET_FEATURE_LABELS,
        feature_columns=feature_columns,
        hidden_units=params.hidden_units,
        warm_start_from=params.warm_start_from,
        config=run_config
    )
    
    return estimator


# Create serving input function for model serving
def _serving_input_receiver_fn(transform_output, schema):
    
    raw_feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    raw_feature_spec.pop(TARGET_FEATURE_NAME)
    raw_feature_spec.pop(WEIGHT_FEATURE_NAME)
    
#     # Create the interface for the serving function with the raw features
#     serving_input_receiver = tf.estimator.export.build_parsing_serving_input_receiver_fn(
#           raw_feature_spec, default_batch_size=None)()
    
#     # Apply the transform function 
#     transformed_features = transform_output.transform_raw_features(
#         serving_input_receiver.features)
    
#     return tf.estimator.export.ServingInputReceiver(
#         transformed_features, 
#         serving_input_receiver.features
#     )

    # Create the interface for the serving function with the raw features
    raw_features = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec)().features

    receiver_tensors = {
        feature: tf.placeholder(shape=[None], dtype=raw_features[feature].dtype) 
        for feature in raw_features
    }

    receiver_tensors_expanded = {
        tensor: tf.reshape(receiver_tensors[tensor], (-1, 1)) 
        for tensor in receiver_tensors
    }

    # Apply the transform function 
    transformed_features = transform_output.transform_raw_features(
        receiver_tensors_expanded)

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, receiver_tensors)


# Create eval input function for model evaluation
def _eval_input_receiver_fn(transform_output, schema):
    
    raw_feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    serving_input_receiver = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)()
    
    features = serving_input_receiver.features.copy()
    transformed_features = transform_output.transform_raw_features(features)

    features.update(transformed_features)
    
    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=serving_input_receiver.receiver_tensors,
        labels=transformed_features[TARGET_FEATURE_NAME]
    )



# TFX will call this function
def trainer_fn(hparams, schema):

    hidden_units = [128, 64]
    batch_size = 40
    train_steps = hparams.train_steps
    eval_steps =  hparams.eval_steps
    model_dir = hparams.serving_model_dir
    
    hparams.hidden_units = hidden_units
    
    # Load TFT Transform Output
    transform_output = tft.TFTransformOutput(hparams.transform_output)
    
    # Create Train Sepc
    train_spec = tf.estimator.TrainSpec(
        input_fn = make_input_fn(
            hparams.train_files,
            transform_output,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True
      ),
      max_steps=train_steps
    )
    
    # Create Exportor with the serving receiver function
    serving_receiver_fn = lambda: _serving_input_receiver_fn(transform_output, schema)
    exporter = tf.estimator.FinalExporter('census', serving_receiver_fn)
    
    # Create Eval Sepc
    eval_spec = tf.estimator.EvalSpec(
        input_fn = make_input_fn(
            hparams.eval_files,
            transform_output,
            batch_size=batch_size
        ),
        exporters=[exporter],
        start_delay_secs=0,
        throttle_secs=0,
        steps=eval_steps,
        name='census-eval'
    )
    
    # Create Run Config
    run_config = tf.estimator.RunConfig(
        tf_random_seed=19831006,
        save_checkpoints_steps=200, 
        keep_checkpoint_max=3, 
        model_dir=model_dir,
        log_step_count_steps=10
    )
    
    # Create estimator
    estimator = create_estimator(hparams, transform_output, run_config)

    # Create an input receiver for TFMA processing
    eval_receiver_fn = lambda: _eval_input_receiver_fn(
        transform_output, schema)
    
    return {
        'estimator': estimator,
        'train_spec': train_spec,
        'eval_spec': eval_spec,
        'eval_input_receiver_fn': eval_receiver_fn
    }
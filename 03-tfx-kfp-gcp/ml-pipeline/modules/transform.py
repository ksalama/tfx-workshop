
import tensorflow_transform as tft
import tensorflow_data_validation as tfdv

TARGET_FEATURE_NAME = 'income_bracket'
WEIGHT_FEATURE_NAME = 'fnlwgt'
RAW_SCHEMA_LOCATION = './raw_schema/schema.pbtxt'

raw_schema = tfdv.load_schema_text(RAW_SCHEMA_LOCATION)

def preprocessing_fn(input_features):

    processed_features = {}

    for feature in raw_schema.feature:
        
        # Pass the target feature as is.
        if feature.name in [TARGET_FEATURE_NAME, WEIGHT_FEATURE_NAME]:
            processed_features[feature.name] = input_features[feature.name]
            continue

        if feature.type == 1:
            # Extract vocabulary and integerize categorical features.
            processed_features[feature.name+"_integerized"] = tft.compute_and_apply_vocabulary(
                input_features[feature.name], vocab_filename=feature.name)
        else:
            # normalize numeric features.
            processed_features[feature.name+"_scaled"] = tft.scale_to_z_score(input_features[feature.name])

        # Bucketize age using quantiles. 
        quantiles = tft.quantiles(input_features["age"], num_buckets=5, epsilon=0.01)
        processed_features["age_bucketized"] = tft.apply_buckets(
            input_features["age"], bucket_boundaries=quantiles)

    return processed_features
FROM tensorflow/tfx:0.15.0
RUN pip install -U tensorflow-serving-api==1.15 tensorflow==1.15
RUN mkdir ~/workspace
RUN mkdir ~/workspace/modules
COPY modules/train.py ~/workspace/modules/
COPY modules/transform.py ~/workspace/modules/
COPY modules/custom_components.py ~/workspace/modules/
COPY modules/__init__.py ~/workspace/modules/
RUN mkdir ~/workspace/raw_schema
COPY raw_schema/schema.pbtxt ~/workspace/raw_schema/
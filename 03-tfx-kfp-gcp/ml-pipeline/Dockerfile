FROM tensorflow/tfx:0.15.0
RUN pip install -U tensorflow-serving-api==1.15 tensorflow==1.15
RUN mkdir modules
COPY modules/train.py modules/
COPY modules/transform.py modules/
COPY modules/custom_components.py modules/
COPY modules/__init__.py modules/
RUN mkdir raw_schema
COPY raw_schema/schema.pbtxt raw_schema/
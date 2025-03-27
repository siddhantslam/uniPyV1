import fiftyone as fo
from fiftyone import zoo

dataset = zoo.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)
session.wait()
import click


@click.group()
def run():
  pass


@run.command()
# @click.argument('API_KEY', required=True)
@click.option("--api-key", help="Your Roboflow API Key", required=True)
def download_data(api_key: str):
  from roboflow import Roboflow

  rf = Roboflow(api_key=api_key)
  project = rf.workspace("doorswindows").project("door-windows")
  version = project.version(1)
  _ = version.download("yolov9")

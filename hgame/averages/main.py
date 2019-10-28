import click

from . import process
from . import tojson


@click.group()
def cli():
    pass


@cli.command("csv")
@click.argument("source")
def do_csv(source):
    process.process_source(source)


@cli.command("json")
@click.argument("source")
def do_json(source):
    tojson.main(source)

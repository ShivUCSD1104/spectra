import typer

app = typer.Typer()


@app.command()
def inspect(file: str):
    """Profile every tensor in an artifact."""
    print("not yet implemented")

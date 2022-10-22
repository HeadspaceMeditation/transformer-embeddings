from nox_poetry import Session, session


@session(python=["3.7", "3.8", "3.9", "3.10"])
def tests(session: Session):
    """Run the tests and generate reports."""
    session.run_always("poetry", "install", external=True)
    session.run(
        "pytest",
        "--log-cli-level=20",
    )

import requests


def test_events():
    url = (
        "https://api.github.com/repos/quantumlib/cirq/issues/3227/events?per_page=100"
    )
    response = requests.get(url)
    print(response.headers)

import requests

from nrw_connection_risk.collector.client import RateLimiter, TimetablesClient


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_rate_limiter_never_exceeds_limit_in_any_window():
    clock = FakeClock()
    rl = RateLimiter(max_calls=5, window_s=60, clock=clock, sleep=clock.sleep)
    times = []
    for _ in range(12):
        rl.acquire()
        times.append(clock.t)
    for i, t in enumerate(times):
        assert sum(1 for u in times if t <= u < t + 60) <= 5
    assert times[5] >= 60          # the 6th call had to wait a full window


class FakeResp:
    def __init__(self, status, text="<timetable/>", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = 0

    def get(self, url, timeout):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_client(responses):
    clock = FakeClock()
    session = FakeSession(responses)
    client = TimetablesClient("https://x", "id", "key", session=session,
                              limiter=RateLimiter(50, clock=clock, sleep=clock.sleep),
                              sleep=clock.sleep, max_retries=3)
    return client, session


def test_sets_auth_headers():
    client, session = make_client([])
    assert session.headers["DB-Client-Id"] == "id" and session.headers["DB-Api-Key"] == "key"


def test_retries_on_5xx_then_succeeds():
    client, session = make_client([FakeResp(503), FakeResp(200, "<timetable station='X'/>")])
    r = client.rchg("8000207")
    assert r is not None and r.status == 200 and session.calls == 2
    assert r.collected_at.tzinfo is not None


def test_retries_on_connection_error():
    client, session = make_client([requests.ConnectionError("down"), FakeResp(200)])
    assert client.fchg("8000207") is not None and session.calls == 2


def test_gives_up_after_max_retries_without_raising():
    client, session = make_client([FakeResp(500)] * 3)
    assert client.fchg("8000207") is None and session.calls == 3 and client.errors_total == 1


def test_does_not_retry_on_401():
    client, session = make_client([FakeResp(401, "unauthorized")])
    assert client.rchg("8000207") is None and session.calls == 1

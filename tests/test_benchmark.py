import pytest

import benchmark


@pytest.mark.parametrize("robot", benchmark.ROBOTS)
def test_no_adaptation(robot):
  no_adaptation = benchmark.make('no_adaptation', robot)
  assert len(list(no_adaptation.test_tasks())) == 0
  assert len(list(no_adaptation.train_tasks())) == len(benchmark._TASKS)

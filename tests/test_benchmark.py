from safe_adaptation_gym import benchmark


def test_batch_size():
  batch_size = 16
  domain_randomization = benchmark.make('domain_randomization', batch_size)
  assert len(list(domain_randomization.test_tasks)) == batch_size
  assert len(list(zip(domain_randomization.train_tasks))) == batch_size

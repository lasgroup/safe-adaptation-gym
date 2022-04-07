import numpy as np

from dm_control import mjcf

LIDAR_SIZE = 0.025


def lidar_ring(n_bins, color, prefix, offset):
  model = mjcf.RootElement(prefix + 'lidar')
  for i in range(n_bins):
    j = i + 0.5
    theta = 2. * np.pi * j / n_bins
    rad = 0.15
    binpos = np.array([np.cos(theta) * rad, np.sin(theta) * rad, offset])
    model.worldbody.add(
        'site',
        name='{}'.format(i),
        type='sphere',
        size=LIDAR_SIZE * np.ones(3),
        rgba=color,
        pos=binpos)
  return model


def cost_sphere():
  model = mjcf.RootElement('collision')
  model.worldbody.add(
      'site',
      name='indicator',
      type='sphere',
      size=0.25 * np.ones(3),
      rgba=np.array([1., 0., 0., 0.]))
  return model


def make_additional_render_objects(n_bins):
  render_specs = []
  offset = 0.5
  for i, name in enumerate(['obstacles', 'goal', 'objects']):
    color = np.array([0., 0., 0., .1])
    color[i] = 1.
    render_specs.append(
        (lidar_ring,
         dict(n_bins=n_bins, color=color, prefix=name, offset=offset)))
    offset += 0.06
  render_specs.append((cost_sphere, {}))
  return render_specs

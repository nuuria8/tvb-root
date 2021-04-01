from tvb.simulator.lab import *

import numpy as np
import math

import matplotlib.pyplot as plt

from tvb.rateML.XML2model import RateML

class regularRun:

	def __init__(self, sim_length, g, s, dt, period, omega = 60, filename='connectivity_68.zip'):
		self.sim_length = sim_length
		self.g = np.array([g])
		self.s = np.array([s])
		self.dt = dt
		self.period = period
		self.omega = omega * 2.0 * math.pi / 1e3
		(self.connectivity, self.coupling) = self.tvb_connectivity(filename)
		self.SC = self.connectivity.weights
		

	def tvb_connectivity(self, filename):
		white_matter = connectivity.Connectivity.from_file(source_file=filename)
		white_matter.configure()
		white_matter.speed = np.array([self.s])
		white_matter_coupling = coupling.Linear(a=self.g)
		return white_matter, white_matter_coupling
	
	def tvb_python_model(self, modelExec):
		# populations = models.Generic2dOscillator()	# original
		# populations = models.KuramotoT()				# generated
		# populations = models.OscillatorT()			# generated
		# populations = models.MontbrioT()				# generated
		# populations = models.RwongwangT()				# generated
		# populations = models.EpileptorT()				# generated
		model = 'models.' + modelExec + '()'
		populations = eval(model)
		populations.configure()
		populations.omega = np.array([self.omega])
		return populations


	def simulate_python(self, modelExec):
		# Initialize Model
		model = self.tvb_python_model(modelExec)
		# Initialize integrator
		# integrator = integrators.EulerDeterministic(dt=self.dt)
		integrator = integrators.EulerStochastic(dt=1, noise=noise.Additive(nsig=np.array([1e-5])))
		# Initialize Monitors
		monitorsen = (monitors.TemporalAverage(period=self.period))
		# Initialize Simulator
		sim = simulator.Simulator(model=model, connectivity=self.connectivity,
								  coupling=coupling.Linear(a=np.array([0.5 / 50.0])), integrator=integrator,
									  monitors=[monitorsen])
		sim.configure()
		(time, data) = sim.run(simulation_length=self.sim_length)[0]

		print(data.shape)
		plt.plot((data[:, 0, :, 0]), 'k', alpha=.2)
		plt.show()

		return (time, data)

if __name__ == '__main__':

	model_filename = 'Oscillator'
	language='python'

	RateML(model_filename, language)

	simtime = 5e2
	g = 0.1
	g = 0.0042
	s = 4.0
	dt = 1
	period = 1

	model = 'OscillatorT'
	# model='Generic2dOscillator'
	regularRun(simtime, g, s, dt, period).simulate_python(model)
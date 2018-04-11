import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns
import random


def jp_plotData(dataFrame, titleString, saveToFile='', nsamples = 100, vars=None, hue=None):
	"""Generate plots of the data in dataframe. Right now only joinplots are generated"""

	#plot nsamples random elements from the sample
	random_elements = random.sample(range(dataFrame.shape[0]), nsamples)
	dataFrame_plot = dataFrame.iloc[random_elements]

	print(dataFrame_plot.shape)

	if vars != None:
		p = sns.pairplot(dataFrame_plot, hue=hue, diag_kind='kde', vars=vars, dropna=True)
	else:
		p = sns.pairplot(dataFrame_plot, hue=hue, diag_kind='kde', dropna=True)

	plt.suptitle(titleString, y = 1.025)

	if saveToFile != '':
		plt.savefig(saveToFile, format='png', pad_inches=0.5, bbox_inches='tight')
		plt.close()
	else:
		plt.show()

def scatterplot_fault(dataFrame, titleString, saveToFile='', nsamples = 100):

	n_m = dataFrame.shape[0]
	random_elements = random.sample(range(n_m), nsamples)
	df_plot = dataFrame.iloc[random_elements]

	desiredFeatures = df_plot.columns[:2]

	plot_xy = df_plot.values
	plot_c = df_plot['selectedFault'].values.astype(float)
	plot_abnormal = plot_xy[plot_c != 20]
	plot_normal = plot_xy[plot_c == 20]

	plt.scatter(plot_normal[:,0], plot_normal[:,1], c = 'blue', cmap='inferno', label='Normal observations')
	plt.scatter(plot_abnormal[:,0], plot_abnormal[:,1], c = 'red', cmap='inferno', label='Faulty observations')
	plt.legend()
	plt.title(titleString)
	plt.xlabel(desiredFeatures[0])
	plt.ylabel(desiredFeatures[1])

	if saveToFile != '':
		plt.savefig(saveToFile, format='png', pad_inches=0.5, bbox_inches='tight')
		plt.close()
	else:
		plt.show()


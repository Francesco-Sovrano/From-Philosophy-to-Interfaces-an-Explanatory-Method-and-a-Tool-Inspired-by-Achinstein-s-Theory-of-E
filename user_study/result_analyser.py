import csv
import numpy as np
import json
import scipy.stats as scipy_stats
from distribution_fit_lib import best_fit_distribution, get_params_description

GENDER_LIST = [
	# 'Male',
	# 'Female',
	'All'
]
FINANCE_EXP_LIST = [
	'Yes',
	'No'
]
KEEP_REJECTED = False

is_empty = lambda x: x==''
is_invalid_answer = lambda x: is_empty(x) or x=='No'

format_sus = lambda sus: list(map(lambda x: int(x) if x!='' else 3, sus))
format_answers = lambda answers: list(map(lambda x: 1 if x=='Yes' else 0, answers))
format_time = lambda time_str: tuple(map(int,time_str.split(':')[:2]))

def time_to_seconds(time):
	h,m = time
	return 60*60*h+60*m

def compute_sus_score(sus):
	x = sum(sus[::2]) - 5
	y = 25 - sum(sus[1::2])
	return (x + y)*2.5

def get_stat_dict(value_list):
	# print(value_list)
	return {
		'median': np.median(value_list),
		'mean': np.mean(value_list),
		'std': np.std(value_list),
		# 'max': max(value_list),
		# 'min': min(value_list),
	}

row_list = []
with open('data/questionnaire_alternative.csv', newline='') as csvfile:
	row_list += list(map(lambda x: ['Yes']+x, csv.reader(csvfile, delimiter=',')))
with open('data/questionnaire_baseline.csv', newline='') as csvfile:
	row_list += list(map(lambda x: ['No']+x, csv.reader(csvfile, delimiter=',')))

experiment_sus_dict = {
	'Yes': {g:[] for g in GENDER_LIST},
	'No': {g:[] for g in GENDER_LIST},
}

INTRO_QUESTIONS = 7
EFFECTIVENESS_QUESTIONS = 7
END_QUESTIONS = 2
SUS_QUESTIONS = 10
for i,row in enumerate(row_list[1:]):
	# Timestamp, Gender, Age, Do you have experience with Credit Approval Systems or Finance, What browser are you using?, What time is it NOW?
	plugin = row[0]
	rejected, timestamp, gender, age, finance_experience, browser, start_time_str = row[1:1+INTRO_QUESTIONS]
	anwers_and_questions = row[1+INTRO_QUESTIONS:1+INTRO_QUESTIONS+EFFECTIVENESS_QUESTIONS*2]
	end_time_str, suggestion = row[1+INTRO_QUESTIONS+EFFECTIVENESS_QUESTIONS*2:1+INTRO_QUESTIONS+EFFECTIVENESS_QUESTIONS*2+END_QUESTIONS]
	sus = row[1+INTRO_QUESTIONS+EFFECTIVENESS_QUESTIONS*2+END_QUESTIONS:]
	
	if not KEEP_REJECTED and rejected=='Yes':
		continue
	# Filter by previous experience
	if finance_experience not in FINANCE_EXP_LIST:
		continue
	
	end_time = format_time(end_time_str)
	start_time = format_time(start_time_str)
	elapsed_seconds = time_to_seconds(end_time) - time_to_seconds(start_time)
	# Q1 - Correct, Q1 - What automated process was used by the Bank to decide whether to give a loan?, Q2 - Correct, Q2 - What are the known issues of the automated processes (of the specific Credit Approval System) used by the Bank?, Q3 - Correct, Q3 - What did the Credit Approval System decide for Customer 25?, Q4 - Correct, Q4 - What is the "Average age of accounts in months" in this specific context?, Q5 - Correct, Q5 - What is the value of "Average age of accounts in months" that the Bank associated to Customer 25?, Q6 - Correct, Q6 - What should Customer 25 do in order to get its loan application accepted?, Q7 - Correct, Q7 - What is the smallest change to the "Average age of accounts in months" that Customer 25 should do in order to have its loan application accepted by the Bank?
	
	answers = anwers_and_questions[::2]
	# answers = [answers[0], answers[2], answers[3]] # only answers 1, 3, 4
	# Ignore if all answers are empty/nonsensical
	if len(list(filter(is_empty, answers))) == len(answers):
		continue
	# 1- I think that I would like to use these explanations frequently., 2- I found the explanations unnecessarily complex., 3- I thought the explanations were clear to understand., 4- I think that I would need the support of an expert to be able to understand the explanations of the system., 5- I found the various explanation bits in this system were well integrated., 6- I thought there was too much inconsistency in the explanation bits., 7- I would imagine that most people would learn to understand these explanations very quickly., 8- I found the explanations very cumbersome to understand., 9- I felt very confident understanding the explanations., 10- I needed to learn a lot of things before I could get going with these explanations.
	
	
	# Ignore if SUS is empty
	if len(list(filter(is_empty, sus))) > 0:
		continue

	answers = format_answers(answers)
	sus = format_sus(sus)
	sus_score = compute_sus_score(sus)
	# print(i, plugin, sus_score)
	row_dict = {
		'Elapsed Seconds': elapsed_seconds,
		'Effectiveness': sum(answers),
		'Satisfaction': sus_score,
		'scale': sus
	}
	if gender.capitalize() in experiment_sus_dict[plugin]:
		experiment_sus_dict[plugin][gender].append(row_dict)
	experiment_sus_dict[plugin]['All'].append(row_dict)

result_dict = {
	'Yes': {g:[] for g in GENDER_LIST},
	'No': {g:[] for g in GENDER_LIST},
}
for gender in GENDER_LIST:
	for key, sus_dict in experiment_sus_dict.items():
		score_list = list(map(lambda x: x['Satisfaction'], sus_dict[gender]))
		scale_list = list(map(lambda x: x['scale'], sus_dict[gender]))
		efficacy_list = list(map(lambda x: x['Effectiveness'], sus_dict[gender]))
		seconds_list = list(map(lambda x: x['Elapsed Seconds'], sus_dict[gender]))

		key_result_dict = {
			'test_count': len(score_list),
			'Elapsed Seconds': get_stat_dict(seconds_list),
			'Satisfaction': get_stat_dict(score_list),
			'Effectiveness': get_stat_dict(efficacy_list),
		}

		key_result_dict['question_dict'] = {}
		median_sus = []
		for e,q_list in enumerate(zip(*scale_list)):
			key_result_dict['question_dict'][e] = get_stat_dict(q_list)
			median_sus.append(key_result_dict['question_dict'][e]['median'])
		key_result_dict['median_score'] = compute_sus_score(median_sus)
		result_dict[key][gender] = key_result_dict

print('stats:', json.dumps(result_dict, indent=4))

#This test can be used to investigate whether two independent samples were selected from populations having the same distribution.
'''
A low pvalue implies that .
A high pvalue implies that Elapsed Seconds in "No" are not statistically greater than Elapsed Seconds in "Yes".
'''
def test_hypothesis(a, b):
	a_value, a_label = a
	b_value, b_label = b
	# params_dict = {}
	# sse_dict = {}
	# for distr, params, sse in best_fit_distribution(a_value):
	# 	sse_dict[distr] = sse
	# 	params_dict[distr] = [params]
	# for distr, params, sse in best_fit_distribution(b_value):
	# 	if distr not in sse_dict:
	# 		continue
	# 	sse_dict[distr] += sse
	# 	params_dict[distr].append(params)
	# best_distribution = sorted(sse_dict.items(), key=lambda x:x[-1])[0][0]
	# fit_params_a, fit_params_b = params_dict[best_distribution]
	alternatives = ['two-sided','less','greater']
	mannwhitneyu_dict = {}
	for alternative in alternatives:
		mannwhitneyu_dict[b_label + ' is ' + alternative] = scipy_stats.mannwhitneyu(a_value, b_value, use_continuity=True, alternative=alternative)
	return {
		# 'wilcoxon': scipy_stats.wilcoxon(a_value,b_value), # The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution. In particular, it tests whether the distribution of the differences x - y is symmetric about zero. It is a non-parametric version of the paired T-test.
		# 'best_fit_distribution': best_distribution.name,
		# 'params': {
		# 	'a': get_params_description(best_distribution, fit_params_a),
		# 	'b': get_params_description(best_distribution, fit_params_b)
		# },
		'mannwhitneyu': mannwhitneyu_dict,
		'kruskal': scipy_stats.kruskal(a_value,b_value), # Due to the assumption that H has a chi square distribution, the number of samples in each group must not be too small. A typical rule is that each sample must have at least 5 measurements.
	}

for gender in GENDER_LIST:
	# follows loglaplace distribution
	print('Elapsed Seconds', json.dumps(test_hypothesis( # A low mannwhitneyu pvalue (<0.05) implies that Elapsed Seconds in 'No' are statistically greater than Elapsed Seconds in 'Yes'
		(list(map(lambda x: x['Elapsed Seconds'], experiment_sus_dict['No'][gender])),'No'),
		(list(map(lambda x: x['Elapsed Seconds'], experiment_sus_dict['Yes'][gender])),'Yes'),
	), indent=4))

	# follows gennorm distribution
	print('Effectiveness', json.dumps(test_hypothesis( # A low mannwhitneyu pvalue (<0.05) implies that Effectiveness in 'No' are statistically lower than Effectiveness in 'Yes'
		(list(map(lambda x: x['Effectiveness'], experiment_sus_dict['No'][gender])),'No'),
		(list(map(lambda x: x['Effectiveness'], experiment_sus_dict['Yes'][gender])),'Yes'),
	), indent=4))

	# follows dgamma distribution
	print('Satisfaction', json.dumps(test_hypothesis( # A high pvalue (>0.95) implies that 'Yes' and 'No' have very similar scores
		(list(map(lambda x: x['Satisfaction'], experiment_sus_dict['No'][gender])),'No'),
		(list(map(lambda x: x['Satisfaction'], experiment_sus_dict['Yes'][gender])),'Yes'),
	), indent=4))

	a = map(lambda x: x['scale'], experiment_sus_dict['No'][gender])
	b = map(lambda x: x['scale'], experiment_sus_dict['Yes'][gender])
	print('Single SUS scales:')
	sus_scale_dict = {}
	for e,(a_list,b_list) in enumerate(zip(zip(*a),zip(*b))):
		sus_scale_dict[int(e)+1] = test_hypothesis(
			(a_list,'No'),
			(b_list,'Yes'),
		)
	print(json.dumps(sus_scale_dict, indent=4))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plot_list = ['Satisfaction','Effectiveness','Elapsed Seconds']

df_list = []
for gender in GENDER_LIST:
	df_yes = pd.DataFrame(experiment_sus_dict['Yes'][gender])
	df_yes = pd.melt(df_yes, value_vars=plot_list)
	df_yes['user-centred'] = True
	df_yes['gender'] = gender
	df_no = pd.DataFrame(experiment_sus_dict['No'][gender])
	df_no = pd.melt(df_no, value_vars=plot_list)
	df_no['user-centred'] = False
	df_no['gender'] = gender
	df_list += [df_yes,df_no]
df = pd.concat(df_list,ignore_index=True)
print(df.loc[df['variable'] == 'Effectiveness'])

sns.set_style("whitegrid")
g = sns.FacetGrid(df, col="variable", row='gender', sharex=False, sharey=False,)
def my_boxplot(**kwargs):
	x = kwargs.pop('x')
	y = kwargs.pop('y')
	data = kwargs.pop('data')
	ax = sns.boxplot(x=x, y=y, data=data, showfliers=kwargs.get('showfliers'), autorange=kwargs.get('autorange'))

	# Calculate number of obs per group & median to position labels
	medians = data.groupby([x])[y].median().values
	# Add it to the plot
	pos = range(len(medians))
	for tick,label in zip(pos,ax.get_xticklabels()):
		ax.text(
			pos[tick], 
			medians[tick], 
			medians[tick], 
			horizontalalignment='center', 
			size='medium', 
			# color='w', 
			weight='bold',
			ha="center", va="center",
			bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8))
		)
	return ax
ax = g.map_dataframe(my_boxplot, x='user-centred', y='value', showfliers=False, autorange=True).set_titles("{row_name} | {col_name}",bbox=dict(boxstyle="round", ec=(0., 0., 0.), fc=(0.9, 0.9, 0.9))).set_axis_labels('user-centred tool','value')
# Iterate thorugh each axis
for ax in g.axes.flat:
    # Make x and y-axis labels slightly larger
    ax.set_xlabel(ax.get_xlabel(), fontsize='x-large', fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize='x-large', fontweight='bold')

# plt.legend()
plt.tight_layout()
plt.savefig('boxplot.png')

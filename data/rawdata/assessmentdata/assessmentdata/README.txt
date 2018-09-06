WSU Study
Activities of Daily Living and Memory in Older Adulthood and Dementia

Description

These files contain data collected as part of a study carried out by
researchers at Washington State University. The purpose of the study was
to learn more about memory abilities and how they relate to activities that
people are involved with everyday, such as cooking and using the telephone.
The study also focused on investigating the use of smart environment
technologies including automated prompts.

A detailed description of the study is found in the included papers
Study participants completed an initial testing session as well as a
questionnaire. On a separate day, the participant came to the WSU CASAS
smart apartment and performed a set of activities.  Sensor data that was
generated during these activities is available as part of this dataset.

Participants were assigned identifiers during the study.  The identifiers
have been randomized before this data was made available to the public.
In addition, information that could provide insights on participant identify,
including age, gender, and education level, have been removed.

Files

sensorlayout.png: This file shows the sensor layout of the apartment in which
activities were performed.

data/*: This directory contains sensor event files, one per participant.
Events within the file are labeled according to the activity (1-24) and,
where available, the activity step that was being performed when the
sensor event was generated.

The sensors can be categorized by:

   Mxx:       motion sensor
   Ixx:       item sensor for selected items in the kitchen
   Dxx:       door sensor
   AD1-A:     burner sensor
   AD1-B:     hot water sensor
   AD1-C:     cold water sensor
   Txx:       temperature sensors
   P001:      whole-apartment electricity usage

documents/activitylist.txt: This file contains a list of the activities and
their corresponding steps.

documents/activityscores.txt: This file contains scores for activities 1-8
and activities 17-24, as assigned by the experimenter who was watching the
activity being performed via a webcam.  Missing scores are indicated by a "-".

documents/diagnosis.txt: This file contains diagnosis information for
each of the study participants.

papers/*: This directory contains papers that provide details of the study
and related data analysis.

* All use of the data must cite the WSU CASAS smart home project.

D. Cook, A. Crandall, B. Thomas, and N. Krishnan. CASAS: A smart home in a box.
IEEE Computer, 2013.
http://eecs.wsu.edu/~cook/pubs/computer12.pdf

* Papers that reference clinical participant assessment or the prompting
technologies employed for activities 9-16 should cite the following papers:

Seelye, A., Schmitter-Edgecombe, M., Cook, D., and Crandall, A.
Naturalistic assessment of everyday activities and smart environment
prompting technologies in mild cognitive impairment subtypes.
Journal of the International Neuropsychological Society, 2013.
http://journals.cambridge.org/action/displayAbstract?fromPage=online&aid=8824248

Das, B., Krishnan, N., and Cook, D. RACOG and wRACOG: Two Gibbs sampling-based
oversampling techniques. IEEE Transactions on Knowledge and Data Engineering,
2013.

* Papers that reference activities 1-8 should also cite the following paper:

Dawadi, P., Cook, D., Parsey, C., Schmitter-Edgecombe, M., and
Schneider, M. An approach to cognitive assessment in smart homes.
KDD workshop on Medicine and Healthcare, 2011.
http://www.eecs.wsu.edu/~cook/pubs/kdd11w1.pdf

* Papers that reference activities 17-24 should also cite the following paper:

Dawadi, P., Cook, D., and Schmitter-Edgecombe, M. Automated cognitive
health assessment using smart home monitoring of complex tasks.
IEEE Transactions on Human-Machine Systems, 2013.
http://eecs.wsu.edu/~cook/pubs/thms12.pdf

* More information is also available in these papers:

Schmitter-Edgecombe, M., McAlister, C., & Weakley, A. (2012). Naturalistic
assessment of everyday functioning in individuals with mild cognitive
impairment: the day out task. Neuropsychology, 26, 631-641.
http://psycnet.apa.org/psycinfo/2012-20171-001/
doi: 10.1037/a0029352

Schmitter-Edgecombe, M., Parsey, C., & Cook, D. (2011). Cognitive correlates of
functional performance in older adults: comparison of self-report, direct
observation and performance-based measures. Journal of the International
Neuropsychological Society, 17, 853-864. 
http://journals.cambridge.org/action/displayAbstract?fromPage=online&aid=8364852

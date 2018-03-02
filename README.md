Set training dataset:
  take any year direct UR daily runoff ratio, use polynomial range of 1-7 to fit the data, (t)
Set test dataset:
  for all other years, determine the model performance

plot polynomial order (x-axis) vs. model performance (y-axis) (mean, with 95% conf. interval bar)

Then add multiple series for watershed scaling exponent

Add slider to modify scaling exponent

==> What scaling exponent produces smallest cost function?

Defining Training, Validation, and Test data sets

https://machinelearningmastery.com/difference-test-validation-datasets/

###  Data flags

A = Partial Day
D = Dry
R = Revised within the last two years
B = Ice Conditions
E = Estimated
P = Partial dry
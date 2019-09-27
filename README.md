# Decision-tree-based-Premier-league-prediction
Designing a Decision tree-based prediction algorithm to decide winning team in Premier league season
Implement an algorithm that builds a 3-level decision tree to predict the outcome of the games Liverpool played in the 2017/2018 premier league season. The training data given in the file “Training_Data.xlsx” is the outcome of all games of all other teams that ended in the win of one of the two competing teams. The decision tree should predict, based on the values of the given attributes, whether the home team or the away team of the game in which Liverpool is playing will win the game. The attributes given in the file for each game are as follows:

Home Team

Away Team

HS: Home Team Shots

AS: Away Team Shots

HST: Home Team Shots on Target

AST: Away Team Shots on Target

HF: Home Team Fouls Committed

AF: Away Team Fouls Committed

HC: Home Team Corners

AC: Away Team Corners

HY: Home Team Yellow Cards

AY: Away Team Yellow Cards

HR: Home Team Red Cards

AR: Away Team Red Cards

Given that the data is numeric, you will need to discretize the data. The root node of the tree should have 2 possible values. The nodes in the two next levels should have 3 possible values. For discretization of the root node, you can do it based on whether the value is above or below the mean value of the attribute. For the nodes of the other two levels, you can discretize the data using equal spacing discretization (uniform).

Requirements:

• Specifying the nodes at each level of the tree that were identified using your algorithm.

• A confusion matrix for the games Liverpool played given in the testing data file “Liverpool.xlsx”.

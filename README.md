### Business Problem
Predicting which class (average, highlighted) players are according to the scores given to the characteristics of the football players watched by the Scouts.

### About Dataset
According to the characteristics of the football players observed in the matches from the data set Scoutium, the football players evaluated by the scouts, It consists of information including the features scored in # and their scores.

**scoutium_attributes.csv**

- **task_response_id :** The set of a scout's evaluations of all players on a team's roster in a match
- **match_id :** The id of the relevant match
- **evaluator_id :** The id of the evaluator(scout)
- **player_id :** The id of the relevant player
- **position_id :** The id of the position played by the relevant player in that match
  
1: Keeper
2: Stopper
3: Right-back
4: Left back
5: Defensive midfielder
6: Central midfielder
7: Right wing
8: Left wing
9: Offensive midfielder
10: Striker

- **analysis_id :** Set of attribute evaluations of a scout for a player in a match
- **attribute_id :** The id of each attribute that players are evaluated on
- **attribute_value :** The value (points) a scout gives to a player's attribute

<br>

**scoutium_potential_labels.csv**

- **task_response_id :** The set of a scout's evaluations of all players on a team's roster in a match
- **match_id :** The id of the relevant match
- **evaluator_id :** The id of the evaluator(scout)
- **player_id :** The id of the relevant player
- **potential_label :** Label that indicates a scout's final decision regarding a player in a match. (target variable)

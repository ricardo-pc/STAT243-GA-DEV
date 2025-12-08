# Genetic Algorithm for Variable Selection


The `GA` package implements a gentic algorithm to perform variable
selection using cross-validated *R*<sup>2</sup> as the fitness metric.

# How to use the function select()

## Inputs

1.  `X` (numpy array or pandas DataFrame): Predictor matrix (n samples
    by p predictors).

2.  `y` (numpy array or pandas DataFrame): Response vector (length n
    samples).

3.  `parent_selection` (str): “rank” for rank-based selection of parents
    or “tournament” for tournament style selection of parents. The
    default is “rank”.

4.  `crossover_type` (str): “single” for single crossover point or
    “double” for double crossover point. The default is “single”.

5.  `penalty` (float): Complexity penalty. Must be between 0 and 1.
    Default is None, which is equivalent to a penalty of 0.

6.  `model_type` (str): “linear” for linear regression, default, “tree”
    for decision tree, or “lasso” for Lasso regression. Default is
    “linear”. See further details below.

7.  `model_params` (dict): Optional input. Model settings for decision
    tree or Lasso regression. Default is None. See further details
    below.

8.  `pop_size` (int): Generation size. Must be a positive integer.
    Default is ~1.5\*p (where p is the number of predictors).

9.  `n_gen` (int): Number of interations/generations. Must be \> 1.
    Default is 100.

10. `mut_rate` (float): Mutation rate. Must be between 0 and 1. Default
    is 0.01 (1%). A mutation rate \> 0.1 is not recommended.

## Output

A dictionary containing:

-   “selected”: indices of the selected variables (using 0 indexing).

-   “R2”: unpenalized cross-validated *R*<sup>2</sup>

-   “R2pen”: penalized cross-validated *R*<sup>2</sup>

# Demonstrations

## Baseball example

Here we apply the genetic algorithm to a variable selection problem for
baseball data. The goal is to find the best subset of predictors to
predict salary.

``` python
baseball_df.head(6)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">salary</th>
<th data-quarto-table-cell-role="th">average</th>
<th data-quarto-table-cell-role="th">obp</th>
<th data-quarto-table-cell-role="th">runs</th>
<th data-quarto-table-cell-role="th">hits</th>
<th data-quarto-table-cell-role="th">doubles</th>
<th data-quarto-table-cell-role="th">triples</th>
<th data-quarto-table-cell-role="th">homeruns</th>
<th data-quarto-table-cell-role="th">rbis</th>
<th data-quarto-table-cell-role="th">walks</th>
<th data-quarto-table-cell-role="th">...</th>
<th data-quarto-table-cell-role="th">rbisperso</th>
<th data-quarto-table-cell-role="th">walksperso</th>
<th data-quarto-table-cell-role="th">obppererror</th>
<th data-quarto-table-cell-role="th">runspererror</th>
<th data-quarto-table-cell-role="th">hitspererror</th>
<th data-quarto-table-cell-role="th">hrspererror</th>
<th data-quarto-table-cell-role="th">soserrors</th>
<th data-quarto-table-cell-role="th">sbsobp</th>
<th data-quarto-table-cell-role="th">sbsruns</th>
<th data-quarto-table-cell-role="th">sbshits</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>3300</td>
<td>0.272</td>
<td>0.302</td>
<td>69</td>
<td>153</td>
<td>21</td>
<td>4</td>
<td>31</td>
<td>104</td>
<td>22</td>
<td>...</td>
<td>1.3000</td>
<td>0.2750</td>
<td>0.0755</td>
<td>17.2500</td>
<td>38.2500</td>
<td>7.7500</td>
<td>320</td>
<td>1.208</td>
<td>276</td>
<td>612</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>2600</td>
<td>0.269</td>
<td>0.335</td>
<td>58</td>
<td>111</td>
<td>17</td>
<td>2</td>
<td>18</td>
<td>66</td>
<td>39</td>
<td>...</td>
<td>0.9565</td>
<td>0.5652</td>
<td>0.0838</td>
<td>14.5000</td>
<td>27.7500</td>
<td>4.5000</td>
<td>276</td>
<td>0.000</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>2500</td>
<td>0.249</td>
<td>0.337</td>
<td>54</td>
<td>115</td>
<td>15</td>
<td>1</td>
<td>17</td>
<td>73</td>
<td>63</td>
<td>...</td>
<td>0.6293</td>
<td>0.5431</td>
<td>0.0562</td>
<td>9.0000</td>
<td>19.1667</td>
<td>2.8333</td>
<td>696</td>
<td>2.022</td>
<td>324</td>
<td>690</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>2475</td>
<td>0.260</td>
<td>0.292</td>
<td>59</td>
<td>128</td>
<td>22</td>
<td>7</td>
<td>12</td>
<td>50</td>
<td>23</td>
<td>...</td>
<td>0.7812</td>
<td>0.3594</td>
<td>0.0133</td>
<td>2.6818</td>
<td>5.8182</td>
<td>0.5455</td>
<td>1408</td>
<td>6.132</td>
<td>1239</td>
<td>2688</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>2313</td>
<td>0.273</td>
<td>0.346</td>
<td>87</td>
<td>169</td>
<td>28</td>
<td>5</td>
<td>8</td>
<td>58</td>
<td>70</td>
<td>...</td>
<td>1.0943</td>
<td>1.3208</td>
<td>0.0384</td>
<td>9.6667</td>
<td>18.7778</td>
<td>0.8889</td>
<td>477</td>
<td>1.038</td>
<td>261</td>
<td>507</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>2175</td>
<td>0.291</td>
<td>0.379</td>
<td>104</td>
<td>170</td>
<td>32</td>
<td>2</td>
<td>26</td>
<td>100</td>
<td>87</td>
<td>...</td>
<td>1.1236</td>
<td>0.9775</td>
<td>0.0758</td>
<td>20.8000</td>
<td>34.0000</td>
<td>5.2000</td>
<td>445</td>
<td>8.338</td>
<td>2288</td>
<td>3740</td>
</tr>
</tbody>
</table>

<p>6 rows × 28 columns</p>
</div>

We use the log of the salary variable as the response variable. And then
we rescale the remaining predictor variables.

``` python
y_baseball = np.log(baseball_df["salary"])          
X_baseball = baseball_df.drop(columns=["salary"])   
X_scaled = StandardScaler().fit_transform(X_baseball)
```

### Linear vs. Lasso regression

First, we run the genetic algorithm using a linear regression model and
a small penalty.

``` python
select(X=X_scaled, y=y_baseball, model_type="linear", penalty=0.01)
```

    {'selected': [0, 2, 5, 7, 9, 12, 13, 14, 15, 23, 24, 25],
     'R2': 0.7887966978691986,
     'R2pen': 0.7843522534247541}

Then, we run the genetic algorithm using a Lasso regression and the same
small penalty. We input our own dictionary of model parameters. We can
see that Lasso results in a similar *R*<sup>2</sup> as the linear but
with fewer parameters, demonstrating the potential benefit of using
Lasso.

``` python
lasso_params = {
    "alpha": 0.1,
    "max_iter": 5000,
    "tol": 1e-4,
    "random_state": 42
}

select(X=X_scaled, y=y_baseball, model_type="lasso", model_params=lasso_params, penalty=0.01)
```

    {'selected': [3, 7, 12, 13],
     'R2': 0.7514445594770588,
     'R2pen': 0.7499630779955773}

### Different ways to produce new generations

We again run the genetic algorithm using a linear regression model. But
we only use 30 generations (as opposed to the default 100).

``` python
select(X=X_scaled, y=y_baseball, model_type="linear", n_gen=30)
```

    {'selected': [0, 2, 7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 22, 24, 25],
     'R2': 0.7892594480361383,
     'R2pen': 0.7892594480361383}

Then, we change the parent_selection from the default “rank” to
“tournament” and the crossover_type from the default “single” to
“double”. We can see that this run results a similar *R*<sup>2</sup> but
with fewer parameters. This demonstrates that these parent selection and
crossover methods help find the best predictors in fewer generations.

``` python
select(X=X_scaled, y=y_baseball, model_type="linear", parent_selection="tournament", crossover_type="double", n_gen=30)
```

    {'selected': [1, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16, 24, 25],
     'R2': 0.7864799131845263,
     'R2pen': 0.7864799131845263}

## Employee satisfaction example

Here we apply the genetic algorithm to a variable selection problem for
employee satisfaction. The goal is to find the best subset of predictors
to predict satisfaction level.

``` python
employee_df.head(6)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">satisfaction_level</th>
<th data-quarto-table-cell-role="th">last_evaluation</th>
<th data-quarto-table-cell-role="th">number_projects</th>
<th data-quarto-table-cell-role="th">avg_montly_hours</th>
<th data-quarto-table-cell-role="th">time_at_company</th>
<th data-quarto-table-cell-role="th">work_accident</th>
<th data-quarto-table-cell-role="th">promotion_last_5years</th>
<th data-quarto-table-cell-role="th">high_salary</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>0.38</td>
<td>0.53</td>
<td>2.0</td>
<td>157.0</td>
<td>3.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>0.80</td>
<td>0.86</td>
<td>5.0</td>
<td>262.0</td>
<td>6.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>0.11</td>
<td>0.88</td>
<td>7.0</td>
<td>272.0</td>
<td>4.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>0.72</td>
<td>0.87</td>
<td>5.0</td>
<td>223.0</td>
<td>5.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>0.37</td>
<td>0.52</td>
<td>2.0</td>
<td>159.0</td>
<td>3.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">5</td>
<td>0.41</td>
<td>0.50</td>
<td>2.0</td>
<td>153.0</td>
<td>3.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
</tbody>
</table>

</div>

First, we run the genetic algorithm using a linear regression model and
all other settings as default. The linear regression results in a
terrible *R*<sup>2</sup> of \< 0.1.

``` python
y_employee = employee_df["satisfaction_level"]
X_employee = employee_df.drop(columns=["satisfaction_level"])

select(X=X_employee, y=y_employee)
```

    {'selected': [0, 1, 3, 4, 5, 6],
     'R2': 0.06124388912570411,
     'R2pen': 0.06124388912570411}

Then, we run the genetic algorithm using a Decision Tree Regression. We
can see that the decision tree results in a much better *R*<sup>2</sup>
(up to ~0.4). Although we still are not getting great performance here,
this example demonstrates the usefulness of the more flexible Decision
Tree Regression.

``` python
select(X=X_employee, y=y_employee, model_type="tree")
```

    {'selected': [0, 1, 2, 3],
     'R2': 0.3992846667679443,
     'R2pen': 0.3992846667679443}

# Further details on input parameters

### `model_type`

-   “linear”: LinearRegression from sklearn.linear_model

-   “lasso”: Lasso from sklearn.linear_model

    -   Lasso Regression is useful for selecting important predictors by
        shrinking unimportant coefficients to zero.

-   “tree”: DecisionTreeRegressor from sklearn.tree

    -   A Decision Tree Regressor can handle nonlinear relationships and
        potential interactions among predictors.

### `model_params`

Default is None, indicating that the function will use the following
settings:

default_tree = { “max_depth”: 5, “min_samples_split”: 2,
“min_samples_leaf”: 5, “random_state”: 42 }

default_lasso = { “alpha”: 0.05, “max_iter”: 5000, “tol”: 1e-4,
“random_state”: 42 }

# Overview of the genetic algorithm

## Setup

We will use an example to help explain the genetic algorithm. Let’s say
we have 6 predictors and the population size (`pop_size`) is set to 4.

## Generation 0

Since we have 6 predictors, each chromosome will have 6 genes/bits. And
since the population size was set to 4 we will have 4 individuals in
each generation. For the starting population, the chromosomes for each
individual are selected purly at random.

<table>
<thead>
<tr>
<th>individual</th>
<th>chromosome</th>
<th>meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td>A</td>
<td>100100</td>
<td>predictors 0,3</td>
</tr>
<tr>
<td>B</td>
<td>011000</td>
<td>predictors 1,2</td>
</tr>
<tr>
<td>C</td>
<td>110110</td>
<td>predictors 0,1,3,4</td>
</tr>
<tr>
<td>D</td>
<td>000101</td>
<td>predictors 3,5</td>
</tr>
</tbody>
</table>

Then, for each individual, we fit a model (linear regression, Lasso
regression, or Decision Tree Regression depending on `model_type`). For
example, for individual A we would fit a model with predictors 0 and 3.

We calculate 10-fold cross-validated *R*<sup>2</sup>. penalized
*R*<sup>2</sup> is calculated as *R*<sup>2</sup> − *λ**f*, where *λ* is
determined by `penalty` and *f* is the fraction of potential predictors
that are selected.

## Parent selection (`parent_selection`)

### Rank-based selection

One parent is selected with probability proportional to rank. The other
parent is selected purly at random.

<table>
<thead>
<tr>
<th>individual</th>
<th><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th>rank</th>
<th>prob of selection</th>
</tr>
</thead>
<tbody>
<tr>
<td>C</td>
<td>0.6</td>
<td>4</td>
<td>4/10</td>
</tr>
<tr>
<td>A</td>
<td>0.5</td>
<td>3</td>
<td>3/10</td>
</tr>
<tr>
<td>B</td>
<td>0.4</td>
<td>2</td>
<td>2/10</td>
</tr>
<tr>
<td>D</td>
<td>0.3</td>
<td>1</td>
<td>1/10</td>
</tr>
</tbody>
</table>

### Tournament selection

The set of chromosomes in the generation are randomly partitioned into k
groups. The best individual in each group is chosen as a parent. Games
continue until a sifficient number of parents have been selected.
Parents are then paried randomly for breeding. Tournament selection
applies more selective pressure than rank-based selection.

## Parent crossover (`crossover_type`)

### Single-point crossover

For each pair of parents, a random position is selected to split the
chromosomes. The left chromosome segment from one parent is glued to the
right chromosome segment from the other parent to form a child. The
remaining segments are combined to form a second child.

For example, if parents B and C have been paired for crossover and 3 is
the random position selected:

Parent C: (**110**110)

Parent B: (011**000**)

Child 1: (**110000**)

Child 2: (011110)

### Double-point crossover

Double-point crossover is very similar in concept to single-point, but
two crossover points are chosen.

Parent C: (**11**01**10**)

Parent B: (01**10**00)

Child 1: (**111010**)

Child 2: (010100)

## Mutation

Mutation happens after parent breeding. Each gene/bit has an independent
probability of mutating (set by the user via `mut_rate`).

For example, take child 1 from above and let mutation flip gene 1:

(1**1**0000) → (1**0**0000)

## Ending

This process is continued over however many generations have been
specified via `n_gen`.

# Team member contributions

-   **Ricardo Castillo:** Ricardo completed the first version of the
    minimally functioning code for the genetic algorithm. He also helped
    some with the README documentation.

-   **Kea Rutherford:** Kea improved and vectorized the code from
    Ricardo and added additional functionality (such as additional model
    fitting methods and input validation). Kea wrote about half of the
    tests. She also wrote most of the final README document.

-   **Lyla Traylor:** Lyla found a couple more places to improve
    vectorization of the code. Lyla also added the Tournament selection
    and double-point crossover functionality. She wrote the other half
    of the tests.

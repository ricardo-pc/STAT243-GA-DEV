# Genetic Algorithm for Variable Selection


The `GA` package implements a genetic algorithm to perform variable
selection, balancing predictive performance and model simplicity through
an evolution inspired search process.

## Overview of select()

### Inputs

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

6.  `model_type` (str): “linear” for linear regression, “tree” for
    decision tree regression, or “lasso” for lasso regression. Default
    is “linear”. See further details below.

7.  `model_params` (dict): Optional input. Model settings for decision
    tree or Lasso regression. Default is None. See further details
    below.

8.  `pop_size` (int): Generation size. Must be a positive integer.
    Default is ~1.5\*p (where p is the number of predictors).

9.  `n_gen` (int): Number of interations/generations. Must be \> 1.
    Default is 100.

10. `mut_rate` (float): Mutation rate. Must be between 0 and 1. Default
    is 0.01 (1%). A mutation rate \> 0.1 is not recommended.

### Output

A dictionary containing:

-   “selected”: indices of the selected variables (using 0 indexing).

-   “R2”: unpenalized cross-validated *R*<sup>2</sup>

-   “R2pen”: penalized cross-validated *R*<sup>2</sup>

<br>

## Demonstrations

### Baseball example

Here we apply the genetic algorithm to a variable selection problem for
baseball data. The goal is to find the best subset of predictors to
predict salary.

``` python
print(baseball_df.head(6))
```

       salary  average    obp  runs  hits  doubles  triples  homeruns  rbis  \
    0    3300    0.272  0.302    69   153       21        4        31   104   
    1    2600    0.269  0.335    58   111       17        2        18    66   
    2    2500    0.249  0.337    54   115       15        1        17    73   
    3    2475    0.260  0.292    59   128       22        7        12    50   
    4    2313    0.273  0.346    87   169       28        5         8    58   
    5    2175    0.291  0.379   104   170       32        2        26   100   

       walks  ...  rbisperso  walksperso  obppererror  runspererror  hitspererror  \
    0     22  ...     1.3000      0.2750       0.0755       17.2500       38.2500   
    1     39  ...     0.9565      0.5652       0.0838       14.5000       27.7500   
    2     63  ...     0.6293      0.5431       0.0562        9.0000       19.1667   
    3     23  ...     0.7812      0.3594       0.0133        2.6818        5.8182   
    4     70  ...     1.0943      1.3208       0.0384        9.6667       18.7778   
    5     87  ...     1.1236      0.9775       0.0758       20.8000       34.0000   

       hrspererror  soserrors  sbsobp  sbsruns  sbshits  
    0       7.7500        320   1.208      276      612  
    1       4.5000        276   0.000        0        0  
    2       2.8333        696   2.022      324      690  
    3       0.5455       1408   6.132     1239     2688  
    4       0.8889        477   1.038      261      507  
    5       5.2000        445   8.338     2288     3740  

    [6 rows x 28 columns]

We use the log of the salary variable as the response variable. And we
rescale the remaining predictor variables.

``` python
y_baseball = np.log(baseball_df["salary"])          
X_baseball = baseball_df.drop(columns=["salary"])   
X_scaled = StandardScaler().fit_transform(X_baseball)
```

<br>

**Linear vs. lasso regression:**

First, we run the genetic algorithm using a linear regression model and
a small penalty.

``` python
result1 = select(X=X_scaled, y=y_baseball, model_type="linear", penalty=0.01)

print(f"Number of selected variables: {len(result1["selected"])}")
print(f"Selected variable indices: {result1["selected"]}")
print(f"Cross-validated R^2: {result1["R2"]}")
print(f"Penalized cross-validated R^2: {result1["R2pen"]}")
```

    Number of selected variables: 11
    Selected variable indices: [0, 2, 7, 9, 12, 13, 14, 15, 23, 24, 26]
    Cross-validated R^2: 0.7814708832249216
    Penalized cross-validated R^2: 0.7773968091508475

Then, we run the genetic algorithm using a lasso regression and the same
small penalty. We input our own dictionary of model parameters. We can
see that lasso results in a similar *R*<sup>2</sup> as the linear but
with fewer parameters. This demonstrates the potential benefit of using
lasso, which selects important predictors by shrinking unimportant
coefficients to zero.

``` python
lasso_params = {
    "alpha": 0.1,
    "max_iter": 5000,
    "tol": 1e-4,
    "random_state": 42
}

result2 = select(X=X_scaled, y=y_baseball, model_type="lasso", 
    model_params=lasso_params, penalty=0.01)

print(f"Number of selected variables: {len(result2["selected"])}")
print(f"Selected variable indices: {result2["selected"]}")
print(f"Cross-validated R^2: {result2["R2"]}")
print(f"Penalized cross-validated R^2: {result2["R2pen"]}")
```

    Number of selected variables: 4
    Selected variable indices: [3, 7, 12, 13]
    Cross-validated R^2: 0.7490525800136867
    Penalized cross-validated R^2: 0.7475710985322052

<br>

**Different ways to produce new generations:**

We again run the genetic algorithm using a linear regression model. But
we only use 30 generations (as opposed to the default 100).

``` python
result3 = select(X=X_scaled, y=y_baseball, model_type="linear", n_gen=30)

print(f"Number of selected variables: {len(result3["selected"])}")
print(f"Selected variable indices: {result3["selected"]}")
print(f"Cross-validated R^2: {result3["R2"]}")
print(f"Penalized cross-validated R^2: {result3["R2pen"]}")
```

    Number of selected variables: 14
    Selected variable indices: [0, 2, 7, 9, 10, 12, 13, 14, 15, 19, 22, 24, 25, 26]
    Cross-validated R^2: 0.7884846997732355
    Penalized cross-validated R^2: 0.7884846997732355

Then, we change the parent_selection from the default “rank” to
“tournament” and the crossover_type from the default “single” to
“double”. We (usually) see that this run results in a similar
*R*<sup>2</sup> but with fewer parameters. This demonstrates that these
parent selection and crossover methods can help find the best predictors
in fewer generations. This makes sense because, for example, tournament
selection applies more selective pressure than rank-based selection.

``` python
result4 = select(X=X_scaled, y=y_baseball, model_type="linear", 
    parent_selection="tournament", crossover_type="double", n_gen=30)

print(f"Number of selected variables: {len(result4["selected"])}")
print(f"Selected variable indices: {result4["selected"]}")
print(f"Cross-validated R^2: {result4["R2"]}")
print(f"Penalized cross-validated R^2: {result4["R2pen"]}")
```

    Number of selected variables: 10
    Selected variable indices: [2, 5, 7, 9, 12, 13, 14, 19, 21, 24]
    Cross-validated R^2: 0.7862923160587727
    Penalized cross-validated R^2: 0.7862923160587727

### Employee satisfaction example

Here we apply the genetic algorithm to a variable selection problem for
employee satisfaction. The goal is to find the best subset of predictors
to predict satisfaction level.

``` python
print(employee_df.head(6))
```

       satisfaction_level  last_evaluation  number_projects  avg_montly_hours  \
    0                0.38             0.53              2.0             157.0   
    1                0.80             0.86              5.0             262.0   
    2                0.11             0.88              7.0             272.0   
    3                0.72             0.87              5.0             223.0   
    4                0.37             0.52              2.0             159.0   
    5                0.41             0.50              2.0             153.0   

       time_at_company  work_accident  promotion_last_5years  high_salary  
    0              3.0            0.0                    0.0          0.0  
    1              6.0            0.0                    0.0          0.0  
    2              4.0            0.0                    0.0          0.0  
    3              5.0            0.0                    0.0          0.0  
    4              3.0            0.0                    0.0          0.0  
    5              3.0            0.0                    0.0          0.0  

First, we run the genetic algorithm using a linear regression model and
all other settings as default. The linear regression results in a
terrible *R*<sup>2</sup>.

``` python
y_employee = employee_df["satisfaction_level"]
X_employee = employee_df.drop(columns=["satisfaction_level"])

result5 = select(X=X_employee, y=y_employee)

print(f"Number of selected variables: {len(result5["selected"])}")
print(f"Selected variable indices: {result5["selected"]}")
print(f"Cross-validated R^2: {result5["R2"]}")
print(f"Penalized cross-validated R^2: {result5["R2pen"]}")
```

    Number of selected variables: 6
    Selected variable indices: [0, 1, 3, 4, 5, 6]
    Cross-validated R^2: 0.06116033942125687
    Penalized cross-validated R^2: 0.06116033942125687

Then, we run the genetic algorithm using a decision tree regression. We
can see that the decision tree results in a much better *R*<sup>2</sup>.
Although we still are not getting great performance here, this example
demonstrates the usefulness of the more flexible decision tree
regression that can handle nonlinear relationships and potential
interactions among predictors.

``` python
result6 = select(X=X_employee, y=y_employee, model_type="tree")

print(f"Number of selected variables: {len(result6["selected"])}")
print(f"Selected variable indices: {result6["selected"]}")
print(f"Cross-validated R^2: {result6["R2"]}")
print(f"Penalized cross-validated R^2: {result6["R2pen"]}")
```

    Number of selected variables: 4
    Selected variable indices: [0, 1, 2, 3]
    Cross-validated R^2: 0.4026976873478969
    Penalized cross-validated R^2: 0.4026976873478969

<br>

## Further details on input parameters

### `model_type`

-   “linear”: uses `LinearRegression` from `sklearn.linear_model`

    -   Performs ordinary least squares estimation.

-   “lasso”: uses `Lasso` from `sklearn.linear_model`

    -   Lasso regression is useful for selecting important predictors by
        shrinking unimportant coefficients to zero.

-   “tree”: uses `DecisionTreeRegressor` from `sklearn.tree`

    -   A decision tree regressor can handle nonlinear relationships and
        potential interactions among predictors.

<br>

### `model_params`

Default is None, indicating that the function will use the following
settings:

default_tree = { “max_depth”: 5, “min_samples_split”: 2,
“min_samples_leaf”: 5, “random_state”: 42 }

default_lasso = { “alpha”: 0.05, “max_iter”: 5000, “tol”: 1e-4,
“random_state”: 42 }

<br>

## Overview of the genetic algorithm

### Setup

We will use an example to help explain the genetic algorithm. Let’s say
we have 6 predictors and the population size (`pop_size`) is set to 4.

<br>

### Generation 0

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

Then, for each individual, we fit a model (linear regression, lasso
regression, or decision tree regression depending on `model_type`). For
example, for individual A we would fit a model with predictors 0 and 3.

We calculate 10-fold cross-validated *R*<sup>2</sup>. Penalized
*R*<sup>2</sup> is calculated as *R*<sup>2</sup> − *λ* \* *f*, where *λ*
is determined by `penalty` and *f* is the fraction of potential
predictors that are selected.

<br>

### Parent selection (`parent_selection`)

**Rank-based selection:**

One parent is selected with probability proportional to rank. The other
parent is selected purly at random.

<table>
<thead>
<tr>
<th>individual</th>
<th><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th>rank</th>
<th>probability of selection</th>
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

<br>

**Tournament selection:**

The set of chromosomes in the generation are randomly partitioned into k
groups. The best individual in each group is chosen as a parent. Games
continue until a sufficient number of parents have been selected.
Parents are then paried randomly for breeding. Tournament selection
applies more selective pressure than rank-based selection.

<br>

### Parent crossover (`crossover_type`)

**Single-point crossover:**

For each pair of the parents, a random position is selected to split the
chromosomes. The left chromosome segment from one parent is glued to the
right chromosome segment from the other parent to form a child. The
remaining segments are combined to form a second child.

For example, if parents B and C have been paired and 3 is the random
position selected:

-   Parent C: (**110***110*) and Parent B: (*011***000**)

-   Child 1: (**110000**) and Child 2: (*011110*)

<br>

**Double-point crossover:**

Double-point crossover is very similar in concept to single-point, but
two crossover points are chosen.

-   Parent C: (**11***01***10**) and Parent B: (*01***10***00*)

-   Child 1: (**111010**) and Child 2: (*010100*)

<br>

### Mutation

Mutation happens after parent breeding. Each gene/bit has an independent
probability of mutating (set by the user via `mut_rate`).

For example, take child 1 from above and let mutation flip gene 1:
(1**1**0000) → (1**0**0000)

<br>

### Continue over generations

This process is continued over however many generations have been
specified via `n_gen`.

<br>

## Citation

This workflow comes from the concepts outline in:

Givens, G.H., & Hoeting, J.A. (2012). Computational statistics (2nd
ed.). Wiley. (See Chapter 3: Combinatorial Optimization.)

<br>

## Team member contributions

-   **Ricardo Castillo:** Ricardo produced the first version of the
    minimal functioning code for the genetic algorithm. He also helped
    some with the README documentation.

-   **Kea Rutherford:** Kea improved and vectorized the code from
    Ricardo and added additional functionality (such as additional model
    fitting methods and validation of user inputs). Kea wrote about half
    of the tests. She also wrote most of the final README document.

-   **Lyla Traylor:** Lyla found a couple more places to improve
    vectorization of the code. Lyla also added the tournament selection
    and double-point crossover functionality. She wrote the other half
    of the tests.

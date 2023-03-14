Activity 7 - Linear Discriminant Analysis
================

## Loading necessary packages

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2 ──
    ## ✔ ggplot2 3.3.6     ✔ purrr   0.3.4
    ## ✔ tibble  3.1.8     ✔ dplyr   1.1.0
    ## ✔ tidyr   1.2.0     ✔ stringr 1.4.1
    ## ✔ readr   2.1.2     ✔ forcats 0.5.2
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
library(tidymodels)
```

    ## ── Attaching packages ────────────────────────────────────── tidymodels 1.0.0 ──
    ## ✔ broom        1.0.0     ✔ rsample      1.1.0
    ## ✔ dials        1.0.0     ✔ tune         1.0.0
    ## ✔ infer        1.0.3     ✔ workflows    1.0.0
    ## ✔ modeldata    1.0.0     ✔ workflowsets 1.0.0
    ## ✔ parsnip      1.0.1     ✔ yardstick    1.0.0
    ## ✔ recipes      1.0.1     
    ## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
    ## ✖ scales::discard() masks purrr::discard()
    ## ✖ dplyr::filter()   masks stats::filter()
    ## ✖ recipes::fixed()  masks stringr::fixed()
    ## ✖ dplyr::lag()      masks stats::lag()
    ## ✖ yardstick::spec() masks readr::spec()
    ## ✖ recipes::step()   masks stats::step()
    ## • Use suppressPackageStartupMessages() to eliminate package startup messages

## Loading the data

``` r
resume <- read.csv("https://www.openintro.org/data/csv/resume.csv")
head(resume)
```

    ##   job_ad_id job_city               job_industry   job_type job_fed_contractor
    ## 1       384  Chicago              manufacturing supervisor                 NA
    ## 2       384  Chicago              manufacturing supervisor                 NA
    ## 3       384  Chicago              manufacturing supervisor                 NA
    ## 4       384  Chicago              manufacturing supervisor                 NA
    ## 5       385  Chicago              other_service  secretary                  0
    ## 6       386  Chicago wholesale_and_retail_trade  sales_rep                  0
    ##   job_equal_opp_employer job_ownership job_req_any job_req_communication
    ## 1                      1       unknown           1                     0
    ## 2                      1       unknown           1                     0
    ## 3                      1       unknown           1                     0
    ## 4                      1       unknown           1                     0
    ## 5                      1     nonprofit           1                     0
    ## 6                      1       private           0                     0
    ##   job_req_education job_req_min_experience job_req_computer
    ## 1                 0                      5                1
    ## 2                 0                      5                1
    ## 3                 0                      5                1
    ## 4                 0                      5                1
    ## 5                 0                   some                1
    ## 6                 0                                       0
    ##   job_req_organization job_req_school received_callback firstname  race gender
    ## 1                    0    none_listed                 0   Allison white      f
    ## 2                    0    none_listed                 0   Kristen white      f
    ## 3                    0    none_listed                 0   Lakisha black      f
    ## 4                    0    none_listed                 0   Latonya black      f
    ## 5                    1    none_listed                 0    Carrie white      f
    ## 6                    0    none_listed                 0       Jay white      m
    ##   years_college college_degree honors worked_during_school years_experience
    ## 1             4              1      0                    0                6
    ## 2             3              0      0                    1                6
    ## 3             4              1      0                    1                6
    ## 4             3              0      0                    0                6
    ## 5             3              0      0                    1               22
    ## 6             4              1      1                    0                6
    ##   computer_skills special_skills volunteer military employment_holes
    ## 1               1              0         0        0                1
    ## 2               1              0         1        1                0
    ## 3               1              0         0        0                0
    ## 4               1              1         1        0                1
    ## 5               1              0         0        0                0
    ## 6               0              1         0        0                0
    ##   has_email_address resume_quality
    ## 1                 0            low
    ## 2                 1           high
    ## 3                 0            low
    ## 4                 1           high
    ## 5                 1           high
    ## 6                 0            low

## LDA

``` r
# Convert received_callback to a factor with more informative labels
resume <- resume %>% 
  mutate(received_callback = factor(received_callback, labels = c("No", "Yes")))

# LDA
library(discrim)
```

    ## 
    ## Attaching package: 'discrim'

    ## The following object is masked from 'package:dials':
    ## 
    ##     smoothness

``` r
lda_years <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS") %>% 
  fit(received_callback ~ log(years_experience), data = resume)

lda_years
```

    ## parsnip model object
    ## 
    ## Call:
    ## lda(received_callback ~ log(years_experience), data = data)
    ## 
    ## Prior probabilities of groups:
    ##         No        Yes 
    ## 0.91950719 0.08049281 
    ## 
    ## Group means:
    ##     log(years_experience)
    ## No               1.867135
    ## Yes              1.998715
    ## 
    ## Coefficients of linear discriminants:
    ##                            LD1
    ## log(years_experience) 1.638023

## Predictions

``` r
predict(lda_years, new_data = resume, type = "prob")
```

    ## # A tibble: 4,870 × 2
    ##    .pred_No .pred_Yes
    ##       <dbl>     <dbl>
    ##  1    0.923    0.0769
    ##  2    0.923    0.0769
    ##  3    0.923    0.0769
    ##  4    0.923    0.0769
    ##  5    0.884    0.116 
    ##  6    0.923    0.0769
    ##  7    0.928    0.0724
    ##  8    0.885    0.115 
    ##  9    0.939    0.0612
    ## 10    0.923    0.0769
    ## # … with 4,860 more rows

``` r
augment(lda_years, new_data = resume) %>% 
  conf_mat(truth = received_callback, estimate = .pred_class)
```

    ##           Truth
    ## Prediction   No  Yes
    ##        No  4478  392
    ##        Yes    0    0

``` r
augment(lda_years, new_data = resume) %>% 
  accuracy(truth = received_callback, estimate = .pred_class)
```

    ## # A tibble: 1 × 3
    ##   .metric  .estimator .estimate
    ##   <chr>    <chr>          <dbl>
    ## 1 accuracy binary         0.920

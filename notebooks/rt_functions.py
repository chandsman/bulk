# !pip install --upgrade pip
# !pip install numpy==1.22
# !pip install numba

# dbutils.library.restartPython()
import numpy as np
import pandas as pd
import time
from numba import vectorize, float64, float32, int64, int32
import math



BIA_FEATURE_COLUMNS = [
  'HOUSEHOLD_ID', 'BPN_ID', 'UPC_ID', # in CosmoDB
  'FREQUENT_SHOPPER_FLAG', # 0 or 1 (household level) --> in CosmoDB
  'LATEST_TXN_DTE', # bpn level --> in CosmoDB
  'DEPT_ID', 'AISLE_ID', 'SHELF_ID', # digital catalog (bpn lelve) --> in CosmoDB
  
  'PERSONAL_CYCLE_BPN', 'PERSONAL_CYCLE_SHELF', 'PERSONAL_CYCLE_AISLE', # bpn level

  # transaction counts in the last 13 months that have this bpn/shelf/aisle and all
  'TXN_COUNT_BPN', 'TXN_COUNT_SHELF', 'TXN_COUNT_AISLE', 'TXN_COUNT_ALL',
  
  'DEPT_RANK', # ranking of the dept
  'AISLE_RANK', # ranking of the aisle
  'SHELF_RANK', # ranking of the shelf in the aisle

  # rankings of the bpn in the aisle (two different ways to rank)
  'AISLE_BPN_RANK_SUB', 'AISLE_BPN_RANK',
  # ranking of the bpn in the shelf
  'SHELF_BPN_RANK',
  # median metrics to be used in assigning SM Flag
  'MED_AISLE_BPN_COUNT',
  'MED_AISLE_SHELF_COUNT',
  'MED_HHID_AISLE_COUNT',
  'MED_SHELF_BPN_COUNT',
  
  #'BPN_RANK_6_MONTH', # to flag 6 month items, use BATCH_BPN_RANK instead
  'BATCH_BPN_RANK',
  # bpn level (number of txns that have this bpn in the past 1,2,3,4,5,6 months)
  'PAST_M1_PURCHASE_COUNT_BPN',
  'PAST_M2_PURCHASE_COUNT_BPN',
  'PAST_M3_PURCHASE_COUNT_BPN',
  'PAST_M4_PURCHASE_COUNT_BPN',
  'PAST_M5_PURCHASE_COUNT_BPN',
  'PAST_M6_PURCHASE_COUNT_BPN',
  'IS_SEASONAL',

  # DS can compute these from other metrics
  # but if that takes too long, then these need to be passed as input
  # aisle level (number of txns that have this aisle in the past 3,6 months)
  #'PAST_M1_PURCHASE_COUNT_SHELF',
  #'PAST_M2_PURCHASE_COUNT_SHELF',
  #'PAST_M3_PURCHASE_COUNT_SHELF',
  #'PAST_M4_PURCHASE_COUNT_SHELF',
  #'PAST_M5_PURCHASE_COUNT_SHELF',
  #'PAST_M6_PURCHASE_COUNT_SHELF',
  #'PAST_M1_PURCHASE_COUNT_AISLE',
  #'PAST_M2_PURCHASE_COUNT_AISLE',
  #'PAST_M3_PURCHASE_COUNT_AISLE',
  #'PAST_M4_PURCHASE_COUNT_AISLE',
  #'PAST_M5_PURCHASE_COUNT_AISLE',
  #'PAST_M6_PURCHASE_COUNT_AISLE',
]



def get_recency(pdf, day, level="all"):
  pdf["RECENCY_BPN"] = (np.datetime64(day) - pdf['LATEST_TXN_DTE'].values.astype('datetime64')) // np.timedelta64(1, 'D')
  pdf["BPN_PUR_90"] = (pdf['RECENCY_BPN'].values <= 90).astype(int)
  pdf["BPN_PUR_30"] = (pdf['RECENCY_BPN'].values <= 35).astype(int)

  is_aisle = 1 if level in ["all", "aisle"] else 0
  is_shelf = 1 if level in ["all", "shelf"] else 0
  if is_aisle:
    pdf['RECENCY_AISLE'] = pdf.groupby('AISLE_ID')['RECENCY_BPN'].transform("min")
  if is_shelf:
    pdf['RECENCY_SHELF'] = pdf.groupby('SHELF_ID')['RECENCY_BPN'].transform("min")

def get_count_by_month(pdf, level='all'):
  num_months = [6, 5, 4, 3, 2, 1]
  is_aisle = 1 if level in ["all", "aisle"] else 0
  is_shelf = 1 if level in ["all", "shelf"] else 0
  if is_shelf:
    group_shelf_obj = pdf.groupby("SHELF_ID")
  if is_aisle:
    group_aisle_obj = pdf.groupby("AISLE_ID")

  for num_month in num_months:
    if is_shelf:
      pdf[f'PAST_M{num_month}_PURCHASE_COUNT_SHELF'] = \
        group_shelf_obj[f"PAST_M{num_month}_PURCHASE_COUNT_BPN"].transform("sum")
    if is_aisle:
      pdf[f'PAST_M{num_month}_PURCHASE_COUNT_AISLE'] = \
        group_aisle_obj[f"PAST_M{num_month}_PURCHASE_COUNT_BPN"].transform("sum")

def n_pur_per_pc_pandas(pdf, level):
  pdf[f"N_PUR_PER_PC_{level}"] = 1
  num_days = [180, 150, 120, 90, 60, 30]
  num_months = [6, 5, 4, 3, 2, 1]
  for i, num_day in enumerate(num_days):
    num_month = num_months[i]
    cond = pdf[f'PERSONAL_CYCLE_{level}'].values <= num_day
    pdf.loc[cond, f"N_PUR_PER_PC_{level}"] = pdf.loc[cond, f'PAST_M{num_month}_PURCHASE_COUNT_{level}']
  pdf.loc[pdf[f"N_PUR_PER_PC_{level}"].values < 1, f"N_PUR_PER_PC_{level}"] = 1

def get_relevancy_pandas(pdf, level="all"):
  for ll in ["BPN", "SHELF", "AISLE"]:
    if level != "all" and ll.lower() != level:
      continue
    n_pur_per_pc_pandas(pdf, ll)
    R, PC, N = pdf[f"RECENCY_{ll}"].values, pdf[f"PERSONAL_CYCLE_{ll}"].values, pdf[f"N_PUR_PER_PC_{ll}"].values
    ratios = R / PC
    quotions = ratios.astype(int)
    decays = np.exp(quotions)
    rel_step = 1 / (1 + (2000*decays)*np.abs(0.99999-1))
    rel_growth = 1 / (1 + (1*decays)*np.abs(ratios-quotions-1)*PC)

    ## default, step_ind = 0 and growth_ind = 1
    step_ind, growth_ind = rel_growth*0, rel_growth*0+1
    step_ind[(quotions > 0) & (ratios-quotions <= 0.25)] = 1
    growth_ind[(quotions > 0) & (ratios-quotions <= 0.25)] = 0

    relevant_org = step_ind * rel_step + growth_ind * rel_growth
    moderate_rel = 0.35*(rel_step-rel_growth) + 0.65*(relevant_org)
    moderate_rel[quotions == 0] = 0
    cond = (moderate_rel >= relevant_org).astype(int)
    relevant = moderate_rel*cond + (1-cond)*relevant_org
    relevant = np.round(N * relevant, 2)
    pdf[f"BUYER_RELEVANCE_{ll}"] = relevant
    if ll == "BPN":
      pdf[f"BUYER_RATIO_THRESHOLD_{ll}"] = 0
      pdf.loc[pdf[f"BUYER_RELEVANCE_{ll}"].values >= 0.09, f"BUYER_RATIO_THRESHOLD_{ll}"] = 1
      pdf.loc[pdf[f"BUYER_RELEVANCE_{ll}"].values >= 0.21, f"BUYER_RATIO_THRESHOLD_{ll}"] = 2

@vectorize([float32(float32, float32),
          float64(float64, float64)], nopython=True, fastmath=True, target='parallel')
def compute_relevancy(ratio, pcycle):
  lookup_lst = (-1.0, 0.9484, 0.8712, 0.7134, 0.4780, 0.2520, 0.1103, 0.0436, 0.0165, 0.0061) # don't use the first element
  if ratio < 0:
    ratio
  quotion = int(ratio) 
  if quotion == 0:
    decay = math.exp(quotion)
    rel_step = 1 / (1 + (2000*decay)*abs(0.99999-1))
    rel_growth = 1 / (1 + (1*decay)*abs(ratio-quotion-1)*pcycle)
    cond = 1 if (quotion > 0) and (ratio-quotion <= 0.25) else 0
    # default, step_ind = 0 and growth_ind = 1
    growth_ind = 1 - cond
    relevant_org = (1-growth_ind)*rel_step + growth_ind*rel_growth
    moderate_rel = 0.35*(rel_step-rel_growth) + 0.65*(relevant_org)
    moderate_rel = 0 if quotion == 0 else moderate_rel
    cond = 1 if moderate_rel >= relevant_org else 0
    relevant_1purchase = moderate_rel*cond + (1-cond)*relevant_org
  elif quotion <= 9:
    relevant_1purchase = lookup_lst[quotion]
  else:
    relevant_1purchase = 0

  return relevant_1purchase

@vectorize([int32(int32, int32, int32, int32, int32, int32, int32),
            int64(int64, int64, int64, int64, int64, int64, int64)], 
           nopython=True, fastmath=True, target='parallel')
def compute_n_pur_per_pc(
  pcycle, 
  onemonth, twomonth, threemonth, 
  fourmonth, fivemonth, sixmonth
):
  if pcycle > 180:
    return 1
  elif pcycle > 150:
    return max(1, sixmonth)
  elif pcycle > 120:
    return max(1, fivemonth)
  elif pcycle > 90:
    return max(1, fourmonth)
  elif pcycle > 60:
    return max(1, threemonth)
  elif pcycle > 30:
    return max(1, twomonth)
  else:
    return max(1, onemonth)

def n_pur_per_pc(pdf, level):
  pdf[f"N_PUR_PER_PC_{level}"] = compute_n_pur_per_pc(
    pdf[f"PERSONAL_CYCLE_{level}"].values,
    pdf[f'PAST_M1_PURCHASE_COUNT_{level}'].values,
    pdf[f'PAST_M2_PURCHASE_COUNT_{level}'].values,
    pdf[f'PAST_M3_PURCHASE_COUNT_{level}'].values,
    pdf[f'PAST_M4_PURCHASE_COUNT_{level}'].values,
    pdf[f'PAST_M5_PURCHASE_COUNT_{level}'].values,
    pdf[f'PAST_M6_PURCHASE_COUNT_{level}'].values
  )

def get_relevancy(pdf, level="all"):
  for ll in ["BPN", "SHELF", "AISLE"]:
    if level != "all" and ll.lower() != level:
      continue
    n_pur_per_pc(pdf, ll)

    R, PC, N = pdf[f"RECENCY_{ll}"].values, pdf[f"PERSONAL_CYCLE_{ll}"].values, pdf[f"N_PUR_PER_PC_{ll}"].values
    ratios = R / PC
    relevant = compute_relevancy(ratios, PC)
    relevant = np.round(N * relevant, 2)
    pdf[f"BUYER_RELEVANCE_{ll}"] = relevant

    if ll == "BPN":
      pdf[f"BUYER_RATIO_THRESHOLD_{ll}"] = 0
      pdf.loc[pdf[f"BUYER_RELEVANCE_{ll}"].values >= 0.09, f"BUYER_RATIO_THRESHOLD_{ll}"] = 1
      pdf.loc[pdf[f"BUYER_RELEVANCE_{ll}"].values >= 0.21, f"BUYER_RATIO_THRESHOLD_{ll}"] = 2

def get_active_seasonal(pdf, active_bpn):
  
  if 'IS_SEASONAL' not in pdf.columns:
    do_compute_seasonal = False
  elif (pdf['IS_SEASONAL'].values == 1).astype(int).sum() > 0:
    do_compute_seasonal = True
  else:
    do_compute_seasonal = False

  if not do_compute_seasonal:
    return do_compute_seasonal

  pdf['IS_PAST_SEASONAL_PURCHASE'] = 0 # meaning neither
  pdf['SEASONAL_SCORE'] = 0
  pdf['SEASONAL_BUCKET'] = 0
  pdf['NDAYS_TILL_PASS_FYEAR'] = pdf['RECENCY_BPN'].values % 365

  pdf.loc[pdf['IS_SEASONAL'].values != 1, "IS_SEASONAL"] = -1 # NOT seasonal, == 1 either seasonal based on model, or based on DEPT_ID
 
  if active_bpn is not None and len(active_bpn) > 0:
    local_bpn_list_pdf = active_bpn.set_index("BPN_ID", inplace=False)
    n_passed_days = local_bpn_list_pdf["NDAYS_PASSED_HOLIDAY"].values[0]
    n_incoming_days = local_bpn_list_pdf["NDAYS_TILL_HOLIDAY"].values[0]
  
    conds = (pdf['IS_SEASONAL'].values == 1) & (np.isin(pdf.index.values, local_bpn_list_pdf.index.values))
    inboth_bpns = np.array(list(set(pdf[conds].index.values).intersection(local_bpn_list_pdf.index.values)))
    pdf.loc[conds, "IS_SEASONAL"] = (1000 + n_incoming_days) # seasonal based on model and active (after this, 1 means seasonal based on DEPT_ID or not currently active)
    pdf.loc[inboth_bpns, "SEASONAL_SCORE"] = local_bpn_list_pdf.loc[inboth_bpns, "SCORE"].values
    del local_bpn_list_pdf

    seasonal_conds = pdf['IS_SEASONAL'].values >= 1000 # seasonal based on model and active
  
    # consider items purchased within 90 days
    recency_conds = pdf['RECENCY_BPN'].values <= 90

    pass_conds = (seasonal_conds) & (recency_conds) & (pdf['RECENCY_BPN'].values >= n_passed_days) # purchase before the prev. holiday
    pdf.loc[pass_conds, "IS_PAST_SEASONAL_PURCHASE"] = 1 # meaning YES
    del pass_conds
  
    # purchase after the prev. holiday but before prev.holiday + 7 days
    pass_conds = (seasonal_conds) & (recency_conds) & (pdf['RECENCY_BPN'].values < n_passed_days) & (pdf['RECENCY_BPN'].values >= max(0, n_passed_days-7))
    if n_passed_days + n_incoming_days >= 8:
      pdf.loc[pass_conds, "IS_PAST_SEASONAL_PURCHASE"] = 1 # meaning YES
    else:
      pdf.loc[pass_conds, "IS_PAST_SEASONAL_PURCHASE"] = -1 # meaning NO
    del pass_conds
  
    pass_conds = (pdf['RECENCY_BPN'].values < max(0, n_passed_days-7)) # purchase more than 7 days after the prev. holiday
    next_conds = (pdf['RECENCY_BPN'].values + n_incoming_days <= 30)   # purchase within 30 days of the next/incomming holiday
    if n_passed_days + n_incoming_days >= 37:
      # for example, ndays between holiday is 37, purchase_dte - prev_holiday = 8 and post_holiday - purchase_dte = 29 ==> then True
      # for example, ndays between holiday is 60, purchase_dte - prev_holiday = 5 and post_holiday - purchase_dte = 55 ==> then False
      time_conds = pass_conds & next_conds
    else:
      time_conds = next_conds

    no_model_conds = (pdf['IS_SEASONAL'].values == 1) & (recency_conds) & (time_conds) & (pdf['PAST_M3_PURCHASE_COUNT_BPN'].values < 2)
    pdf.loc[no_model_conds, "IS_PAST_SEASONAL_PURCHASE"] = -1 # meaning NO
    has_model_conds = (pdf['IS_SEASONAL'].values >= 1000) & (recency_conds) & (time_conds) & (pdf['PAST_M2_PURCHASE_COUNT_BPN'].values < 2)
    pdf.loc[has_model_conds, "IS_PAST_SEASONAL_PURCHASE"] = -1 # meaning NO

  # three cases:
  # 1) active_bpn is empty
  # 2) active_bpn is not empty but does not contain this product (e.g., Valentine candy bought on 2024-02-10 while current date is 2024-02-20)
  # 3) product bought recently but is not in the list of model predicted seasonal products
  recency_conds = pdf['RECENCY_BPN'].values <= 30 # consider items that were purchased within 30 days
  seasonal_conds = pdf['IS_SEASONAL'].values >= 1
  pass_conds = (pdf['IS_PAST_SEASONAL_PURCHASE'].values == 0) & (seasonal_conds) & (recency_conds) & (pdf['PAST_M1_PURCHASE_COUNT_BPN'].values < 2)
  pdf.loc[pass_conds, "IS_PAST_SEASONAL_PURCHASE"] = -1 # meaning NO
  del pass_conds

  # consider items purchased more than 90 days ago
  recency_conds = (pdf['RECENCY_BPN'].values > 90)
  pass_conds = (seasonal_conds) & (recency_conds) # purchased more than 3 months ago
  pdf.loc[pass_conds, "IS_PAST_SEASONAL_PURCHASE"] = 1 # meaning YES
  del pass_conds

  if len(pdf) == (pdf["IS_PAST_SEASONAL_PURCHASE"].values == 0).astype(int).sum():
    do_compute_seasonal = False
  else:
    do_compute_seasonal = True
  return do_compute_seasonal

def previous_year_guardrail_cond(pdf):

  recency_conds = (pdf["RECENCY_BPN"].values >= 180).astype(int)
  lastyear_bpn_cnt = (pdf["TXN_COUNT_BPN"].values - pdf["PAST_M6_PURCHASE_COUNT_BPN"].values)
  lastyear_shelf_cnt = (pdf["TXN_COUNT_SHELF"].values - pdf["PAST_M6_PURCHASE_COUNT_SHELF"].values)
  # purchased more than ONCE last year
  only_freq_bpn_conds = lastyear_bpn_cnt > 1
  # purchased only ONCE but MORE than ONCE in the same shelf last year, and visit same shelf at least ONCE last 6 months
  need_freq_shelf_conds = (lastyear_bpn_cnt == 1) & (lastyear_shelf_cnt > 1) & (pdf["PAST_M6_PURCHASE_COUNT_SHELF"].values >= 1)
  # purchased product only ONCE and only ONCE in the same shelf last year, but shelf's aisle has become more relevant recently
  need_freq_shelf_recent_conds = (lastyear_bpn_cnt == 1) & (lastyear_shelf_cnt == 1) & (pdf["PAST_M3_PURCHASE_COUNT_AISLE"].values >= 2)
  # purchased two years ago, hence we don't have info in the past, but if the shelf's aisle has become more relevant recently, then accept
  for_old_bpn_conds = (pdf["TXN_COUNT_BPN"].values == 0) & (pdf["PAST_M2_PURCHASE_COUNT_AISLE"].values >= 2)

  guardrail_conds = (only_freq_bpn_conds) | (need_freq_shelf_conds)
  guardrail_conds |= (need_freq_shelf_recent_conds)
  guardrail_conds |= (for_old_bpn_conds)

  # if recency >= 180, must use guardrail, else discard guardrail
  guardrail_conds = (recency_conds * guardrail_conds.astype(int) + (1-recency_conds)).astype(int)

  # here 1/True means passing at least one guardrails conditions or just not having any guardrail
  return guardrail_conds.astype(bool)
  


def last_two_month_freq_conds(pdf):
  conds = (pdf['PAST_M2_PURCHASE_COUNT_BPN'].values > 1)
  return conds

def must_have_conds(pdf):
  conds=((pdf['AISLE_BPN_RANK_SUB'].values <= pdf['MED_AISLE_BPN_COUNT'].values) | (pdf['AISLE_BPN_RANK'].values <= pdf['MED_AISLE_BPN_COUNT'].values))
  conds &= (pdf['SHELF_RANK'].values <= 2*pdf['MED_AISLE_SHELF_COUNT'].values)
  conds &= (pdf['AISLE_RANK'].values <= 2*pdf['MED_HHID_AISLE_COUNT'].values)
  return conds

def smart_basket_flag_regular(pdf, is_freq=True, must_have=None):
  if must_have is None:
    must_have = must_have_conds(pdf)
  conds = must_have
  conds &= (pdf['SHELF_BPN_RANK'].values <= pdf['MED_SHELF_BPN_COUNT'].values)  
  conds &= (pdf['PAST_M3_PURCHASE_COUNT_AISLE'].values > 1)
  if is_freq:
    conds &= (pdf['RECENCY_BPN'].values <= 90)
  return conds

def smart_basket_flag_less_than3_txn_dte(pdf, must_have=None):
  if must_have is None:
    must_have = must_have_conds(pdf)
  conds = must_have
  conds &= (pdf['RECENCY_BPN'].values <= 90)
  return conds

def hh_no_smart_basket_flag(pdf, must_have=None):
  if must_have is None:
    must_have = must_have_conds(pdf)
  conds = must_have  
  conds &= (pdf['SHELF_BPN_RANK'].values <= pdf['MED_SHELF_BPN_COUNT'].values)
  return conds

def unpopular_smart_basket_items(pdf, is_freq=True):
  conds = (pdf['TXN_COUNT_BPN'].values == 1)
  if is_freq:
    conds &= (pdf['PAST_M2_PURCHASE_COUNT_BPN'].values == 0)
  return conds

def remove_depts(pdf):
  baby_dept = (pdf['DEPT_ID'].values == '1_1') & (pdf['TXN_COUNT_BPN'].values < 3)
  tobaco_paper_per_dept = (
    (pdf['DEPT_ID'].values == '1_22') | 
    (pdf['DEPT_ID'].values == '1_27') | 
    (pdf['DEPT_ID'].values == '1_18')
  ) & (pdf['RECENCY_BPN'].values < 30)

  return baby_dept | tobaco_paper_per_dept

def freq_shopper_edge_cases(pdf):
    conds = ((pdf['TXN_COUNT_SHELF'].values == 1) & (pdf['TXN_COUNT_ALL'].values >= 3))
    conds |= ((pdf['TXN_COUNT_AISLE'].values == 1) & (pdf['TXN_COUNT_ALL'].values >= 3))
    conds |= (
      (pdf['PAST_M3_PURCHASE_COUNT_BPN'].values == 1) & (pdf['RECENCY_BPN'].values >= 45) & 
      (pdf['TXN_COUNT_ALL'].values > 3)
    )
    conds |= (
      (pdf['TXN_COUNT_SHELF'].values >= 5) & 
      (pdf['MED_SHELF_BPN_COUNT'].values >= 2) & 
      (pdf['TXN_COUNT_BPN'].values <= 2) & 
      (pdf['SHELF_BPN_RANK'].values > 1) & 
      (pdf['TXN_COUNT_ALL'].values > 3)
    )
    return conds

def distinct_bpn_edge_case(pdf):
    conds = ((pdf['PAST_M6_PURCHASE_COUNT_AISLE'].values == 1) | (pdf['TXN_COUNT_BPN'].values == 1))
    conds &= (pdf['TXN_COUNT_ALL'].values > 3)
    return conds
  
def edge_cases(pdf):
    conds=(
      (pdf['MED_AISLE_BPN_COUNT'].values == 1) & 
      ((pdf['TXN_COUNT_BPN'].values / (pdf['TXN_COUNT_AISLE'].values+0.00001)) < 0.3)
    )
    conds |= ((pdf['MED_AISLE_BPN_COUNT'].values == 1) & (pdf['TXN_COUNT_AISLE'].values == 3))
    return conds

def old_items(pdf):
  conds = (pdf['RECENCY_BPN'].values >= 180)
  return conds

def seasonal_sm_cases(pdf):
  sea_score_conds = (pdf['SEASONAL_SCORE'].values >= 500) # this seasonal product is highly active
  sea_score_conds |= (pdf['SEASONAL_SCORE'].values <= -300) # this semi-seasonal product is highly active
  
  mod = pdf['NDAYS_TILL_PASS_FYEAR'].values
  # mod >= 365-7: near full year....less than 7 days is a full year
  # mod <= 7: just passed full year....passed a full year less than 7 days
  near_full_year_cycle = (mod >= 365-7) | (mod <= 7) 

  guardrail_conds = previous_year_guardrail_cond(pdf)

  conds = (pdf['RECENCY_BPN'].values >= 320) & (near_full_year_cycle) # and was purchased around a year ago from current date
  conds &= sea_score_conds
  conds &= guardrail_conds
  return conds

def remove_sm_seasonal_items(pdf):
  # do nothing if IS_PAST_SEASONAL_PURCHASE == 0

  # e.g., bought turkey 3 weeks before Thanksgiving, but today is 1 week before Thanksgiving
  just_bought_recently_for_this_holiday = (pdf["IS_PAST_SEASONAL_PURCHASE"].values == -1) # recent purchase, maybe even relevant for the incoming holiday
  just_bought_recently_for_this_holiday &= (pdf['IS_SEASONAL'].values >= 1000) # (this is correct), has model so can be both seasonal and semi-seasonal
  just_bought_recently_for_this_holiday &= (pdf['PAST_M2_PURCHASE_COUNT_BPN'].values < 2) # less frequently purchase, maybe it is not regular product

  # e.g., bought gift card 3 weeks before Thanksgiving, but today is 1 week before Thanksgiving
  just_bought_recently_for_this_holiday_no_model = (pdf['IS_SEASONAL'].values == 1) # (this is correct), has no model so can be both seasonal and regular
  just_bought_recently_for_this_holiday_no_model &= (pdf['PAST_M3_PURCHASE_COUNT_BPN'].values < 2) # less frequently purchase, maybe it is not regular product
  just_bought_recently_for_this_holiday_no_model &= (pdf['RECENCY_BPN'].values <= 3) # recently purchased (within 3 days)

  # e.g., bought Easter candies around Easter, now it is near June, should not show Easter products
  # some semi-seasonal products are also regular products, hence ignore the seasonal rule for these products if they are frequently purchased
  bought_awhile_ago_for_other_holidays = (pdf["IS_PAST_SEASONAL_PURCHASE"].values == 1) & (pdf['PAST_M2_PURCHASE_COUNT_BPN'].values <= 1)
  bought_awhile_ago_for_other_holidays &= ((pdf['SEASONAL_SCORE'].values >= -200) & (pdf['SEASONAL_SCORE'].values <= 300)) # semi-seasonal has negative score

  return just_bought_recently_for_this_holiday | just_bought_recently_for_this_holiday_no_model | bought_awhile_ago_for_other_holidays
  
def set_smart_basket_values(pdf, do_compute_seasonal=False):

  n_txn_dtes = len(np.unique(pdf["LATEST_TXN_DTE"].values))
  pdf['SMART_BASKET_FLAG'] = 0

  must_have = must_have_conds(pdf)
  if n_txn_dtes > 3:
    # Get Smart Basket Items for frequent / infrequent shopper
    is_frequent_shopper = pdf['FREQUENT_SHOPPER_FLAG'].iloc[0]
    conds = smart_basket_flag_regular(pdf, is_freq=is_frequent_shopper, must_have=must_have)
  else:
    # Get Smart Basket Items for shopper with less than 3 distinct txn
    conds = smart_basket_flag_less_than3_txn_dte(pdf, must_have=must_have)

  pdf.loc[conds, 'SMART_BASKET_FLAG'] = 1
  del conds

  # If the household still doesn't have any smart basket item, then relax the conditions
  if pdf['SMART_BASKET_FLAG'].values.sum() <= 0:
    conds = hh_no_smart_basket_flag(pdf, must_have=must_have)
    pdf.loc[conds, 'SMART_BASKET_FLAG'] = 1
    del conds

  # add more smart basket items based on count
  conds = last_two_month_freq_conds(pdf)
  # after this operation, 
  # SMART_BASKET_FLAG = 3+2 ==> both high ranked and purchased multiple times
  # SMART_BASKET_FLAG = 2   ==> only purchased multiple times
  # SMART_BASKET_FLAG = 1   ==> only high ranked
  pdf['SMART_BASKET_FLAG'] = conds*2 + pdf['SMART_BASKET_FLAG'].values
  pdf.loc[(pdf['SMART_BASKET_FLAG'].values == 3), 'SMART_BASKET_FLAG'] = 5
  del conds

  # seasonality conditions
  if do_compute_seasonal:
    # SMART_BASKET_FLAG = 4 ==> seasonal products
    conds = (seasonal_sm_cases(pdf)) & (pdf['SMART_BASKET_FLAG'].values < 3+2)
    pdf.loc[conds, 'SMART_BASKET_FLAG'] = 4
    del conds

def unset_smart_basket_values(pdf, do_compute_seasonal=False):

  # seasonal products
  non_seasonal_conds = pdf['SMART_BASKET_FLAG'].values != 4
  seasonal_conds = pdf['SMART_BASKET_FLAG'].values == 4

  # these are the SM items that are not purchased more than ONCE in the last 2 months
  # or are not highly ranked
  sm_type1_conds = pdf["SMART_BASKET_FLAG"].values <= 2

  is_frequent_shopper = pdf['FREQUENT_SHOPPER_FLAG'].values[0]
  # Set Smart Basket vales to 0 based on conditions like TXN_COUNT_ALL 
  # and past month bpn purchase count

  conds = unpopular_smart_basket_items(pdf, is_frequent_shopper)
  pdf.loc[conds & sm_type1_conds & non_seasonal_conds, 'SMART_BASKET_FLAG'] = 0
  del conds

  # SM 0 for paper, pet, tobaco, and baby care dept based on recency
  conds = remove_depts(pdf)
  pdf.loc[conds & sm_type1_conds & non_seasonal_conds, 'SMART_BASKET_FLAG'] = 0
  del conds

  # SM 0 based on various edge cases on txn_count shelf/aisle/bpn/all recency
  if is_frequent_shopper:
    conds = freq_shopper_edge_cases(pdf)
    pdf.loc[conds & sm_type1_conds & non_seasonal_conds, 'SMART_BASKET_FLAG'] = 0
    del conds

  # Additional condition if Smart Basket has currently more than 10 items.
  if (sm_type1_conds).astype(int).sum() >= 10:
    conds = distinct_bpn_edge_case(pdf)
    pdf.loc[conds & sm_type1_conds & non_seasonal_conds, 'SMART_BASKET_FLAG'] = 0
    del conds

  conds = edge_cases(pdf)
  pdf.loc[conds & sm_type1_conds & non_seasonal_conds, 'SMART_BASKET_FLAG'] = 0
  del conds

  conds = old_items(pdf)
  pdf.loc[conds & non_seasonal_conds, 'SMART_BASKET_FLAG'] = 0
  del conds

  if do_compute_seasonal:
    # do NOT put conds & non_seasonal_conds in here
    conds = remove_sm_seasonal_items(pdf)
    pdf.loc[seasonal_conds & conds, 'SMART_BASKET_FLAG'] = 0
    del conds




def rank_less_6m(pdf, do_compute_seasonal=False):
  
  ORG_NON_SM_CONDS = pdf["SMART_BASKET_FLAG"].values <= 0 # do NOT modify this variable

  def seasonal_non_sm_cases(df):
    pass_purchase_conds = df["IS_PAST_SEASONAL_PURCHASE"].values == 1 # 1 means YES
    near_holiday_conds = (df['IS_SEASONAL'].values >= 1000) & (df['IS_SEASONAL'].values <= 1000+14)
    not_near_holiday_conds = (df['IS_SEASONAL'].values > 1000+14) & (df['IS_SEASONAL'].values <= 1000+28)
    high_score_conds = (df['SEASONAL_SCORE'].values >= 500) | (df['SEASONAL_SCORE'].values <= -300) # product is highly active
    low_score_conds = (df['SEASONAL_SCORE'].values < 500) & (df['SEASONAL_SCORE'].values >= 350)
    low_score_conds |= ((df['SEASONAL_SCORE'].values > -300) & (df['SEASONAL_SCORE'].values <= -200)) # semi-seasonal products have negative scores

    mod = df['NDAYS_TILL_PASS_FYEAR'].values
    # mod >= 365-7: near full year....less than 7 days is a full year
    # mod <= 7: just passed full year....passed a full year less than 7 days
    near_full_year_cycle = (mod >= 365-7) | (mod <= 7)
    seasonal_bpns = df['IS_SEASONAL'].values >= 1
    last_year_conds = df['RECENCY_BPN'].values > 320

    guardrail_conds = previous_year_guardrail_cond(df)
    negation_guardrail_conds = ~guardrail_conds
  
    # having seasonal model
    high_seasonal_bucket_conds = ORG_NON_SM_CONDS & pass_purchase_conds & guardrail_conds & high_score_conds & near_holiday_conds 
    low_seasonal_bucket_conds = ORG_NON_SM_CONDS & pass_purchase_conds & guardrail_conds & high_score_conds & not_near_holiday_conds 
    more_low_seasonal_bucket_conds = ORG_NON_SM_CONDS & pass_purchase_conds & guardrail_conds & low_score_conds & near_holiday_conds
    very_low_seasonal_bucket_conds = ORG_NON_SM_CONDS & pass_purchase_conds & negation_guardrail_conds & (high_score_conds | low_score_conds) & near_holiday_conds

    # not having seasonal model (hence no model score)
    med_seasonal_bucket_conds = ORG_NON_SM_CONDS & last_year_conds & pass_purchase_conds & near_full_year_cycle & seasonal_bpns & guardrail_conds

    return high_seasonal_bucket_conds, med_seasonal_bucket_conds, low_seasonal_bucket_conds, more_low_seasonal_bucket_conds, very_low_seasonal_bucket_conds

  if do_compute_seasonal:
    # ranking ACTIVE seasonal items outside SM (not yet purchased recently)
    high_bucket, med_bucket, low_bucket, more_low_bucket, very_low_bucket = seasonal_non_sm_cases(pdf)
    pdf.loc[med_bucket, 'SEASONAL_BUCKET'] = 98 # should be first
    pdf.loc[very_low_bucket, 'SEASONAL_BUCKET'] = 10
    pdf.loc[more_low_bucket, 'SEASONAL_BUCKET'] = 16
    pdf.loc[low_bucket, 'SEASONAL_BUCKET'] = 17 
    pdf.loc[high_bucket, 'SEASONAL_BUCKET'] = 99 # should be last

    mod = pdf['NDAYS_TILL_PASS_FYEAR'].values
    # mod >= 365-14: near full year....less than 14 days is a full year
    # mod <= 14: just passed full year....passed a full year less than 14 days
    pdf['IS_NEAR_FYEAR_14DAYS'] = (mod >= 365-14) | (mod <= 14)

    pdf['HAS_MODEL'] = pdf['SEASONAL_SCORE'].values != 0
    # convert negative scores of semi-seasonal products to positive for ranking purpose
    pdf['SEASONAL_SCORE'] = np.abs(pdf['SEASONAL_SCORE'].values)

  six_month_cond = (pdf["RECENCY_BPN"].values < 180)

  #start = time.time()
  #===================================================================ranking items inside SM=======================================================#
  # need to have columns BPN_RANK_FINAL, HOMEPAGE_FLAG available
  # SMART_BASKET_FLAG = 3+2 ==> both high ranked and purchased multiple times
  # SMART_BASKET_FLAG = 2   ==> only purchased multiple times
  # SMART_BASKET_FLAG = 1   ==> only high ranked
  sm_cond = six_month_cond & (pdf["SMART_BASKET_FLAG"].values > 0) & (pdf["SMART_BASKET_FLAG"].values != 4)
  # SMART_BASKET_FLAG = 4 ==> only high relevant seasonal products
  if do_compute_seasonal:
    sm_seasonal_cond = (pdf["SMART_BASKET_FLAG"].values == 4)
    pdf.loc[sm_seasonal_cond, 'SEASONAL_BUCKET'] = 100
    pdf.loc[sm_seasonal_cond & (pdf["BUYER_RATIO_THRESHOLD_BPN"].values == 0), 'BUYER_RATIO_THRESHOLD_BPN'] = 1
    sm_cond |= sm_seasonal_cond

  # must keep SMART_BASKET_BUCKET = 1 for the seasonal products inside SM, otherwise, they get moved outside of SM
  pdf["SMART_BASKET_BUCKET"] = (pdf["SMART_BASKET_FLAG"].values >= 4).astype(int)
  
  # seasonal products in SM will be at the top if do_compute_seasonal is True
  seasonal_bucket_columns = []
  sort_ascending_logic_seasonal_bucket = []
  
  # sort by high/medium/low relevance buckets
  recency_columns = ["BUYER_RATIO_THRESHOLD_BPN", "SMART_BASKET_BUCKET"]
  sort_ascending_logic_recency = [False, False]
  if do_compute_seasonal:
    # seasonal products in SM will be at the top if do_compute_seasonal is True
    seasonal_bucket_columns = ['SEASONAL_BUCKET']
    sort_ascending_logic_seasonal_bucket = [False]
    # sort by high/medium/low relevance buckets, then by seasonal score
    recency_columns = ["BUYER_RATIO_THRESHOLD_BPN", "SEASONAL_SCORE", "SMART_BASKET_BUCKET"]
    sort_ascending_logic_recency = [False, False, False]
  frequency_columns = [
    'PAST_M3_PURCHASE_COUNT_BPN', 'PAST_M6_PURCHASE_COUNT_BPN', 'TXN_COUNT_BPN'
  ]
  sort_ascending_logic_frequency = [False, False, False]
  derived_columns = [
    'AISLE_RANK',
    'BUYER_RELEVANCE_AISLE',
    'SHELF_RANK',
    'BUYER_RELEVANCE_SHELF',
    'SHELF_BPN_RANK',
    'AISLE_BPN_RANK',
    'BUYER_RELEVANCE_BPN',
    'BPN_ID'
  ]
  sort_ascending_logic_derived = [True, False, True, False, True, True, False, True]
  sm_columns = seasonal_bucket_columns + recency_columns + frequency_columns + derived_columns
  sort_ascending_logic_sm = sort_ascending_logic_seasonal_bucket + sort_ascending_logic_recency + sort_ascending_logic_frequency + sort_ascending_logic_derived

  num_sm = sm_cond.astype(int).sum()
  if num_sm > 0:
    sm_rankings = pdf.loc[sm_cond, sm_columns].sort_values(
      by=sm_columns, ascending=sort_ascending_logic_sm
    )
    pdf.loc[sm_rankings.index.tolist(), 'BPN_RANK_FINAL'] = np.arange(num_sm) + 1

  #end = time.time()
  #print(f"sm_rankings execution time in ms is:  {(end - start)*1000}")

  #===================================================================ranking seasonal items outside SM=======================================================#

  # step 1: ranking all products with SEASONAL_BUCKET == 98 or 99, starting from 1
  # for example:
  # BPN, SM_FLAG, SEASONAL_BUCKET, BPN_RANK_FINAL 
  # E  , 0      , 99             , 1
  # F  , 0      , 98             , 2
  
  # step 2: as with realtime no seasonality, removing some SM items from SM basket to keep it lean
  #     previously, keep their rankings, hence they stay above other non-SM products
  # for example:
  # BPN, SM_FLAG, SEASONAL_BUCKET, BPN_RANK_FINAL
  # A  , 1      , 0              , 1
  # B  , 1      , 0              , 2      
  # C  , 1      , 0              , 3
  # D  , 1      , 0              , 4
  # Becomes:
  # BPN, SM_FLAG, SEASONAL_BUCKET, BPN_RANK_FINAL
  # A  , 1      , 0              , 1
  # B  , 1      , 0              , 2      
  # C  , 0      , 0              , 3
  # D  , 0      , 0              , 4
  #     now, push them down and boost the products with SEASONAL_BUCKET == 98 or 99 up, so that these products are right after SM items
  # for example:
  # BPN, SM_FLAG, SEASONAL_BUCKET, BPN_RANK_FINAL
  # A  , 1      , 0              , 1
  # B  , 1      , 0              , 2      
  # E  , 0      , 99             , 3
  # F  , 0      , 98             , 4
  # C  , 0      , 0              , 5
  # D  , 0      , 0              , 6

  #start = time.time()
  num_sea = 0
  if do_compute_seasonal:
    # ranking ACTIVE seasonal items outside SM (not yet purchased recently)
    seasonal_cond = np.isin(pdf['SEASONAL_BUCKET'].values, [98, 99])
    num_sea = seasonal_cond.astype(int).sum()
    
  # only keep top 120 items as SM items, but keep the rankings of these remaining SM items
  if num_sm > 120:
    conds = (pdf['BPN_RANK_FINAL'].values > 120) & (pdf['BPN_RANK_FINAL'].values <= num_sm) & (pdf['SMART_BASKET_FLAG'].values > 0)
    pdf.loc[conds, "SMART_BASKET_FLAG"] = 0
    # this is equivalent to pdf.loc[conds, "BPN_RANK_FINAL"] += num_sea, but faster
    pdf["BPN_RANK_FINAL"] = num_sea*conds + pdf["BPN_RANK_FINAL"].values
    del conds
  
  # keep the rankings of SM items that in bucket BUYER_RATIO_THRESHOLD_BPN = 0 and SMART_BASKET_BUCKET == 0, but move them out of SM
  # they are ranked at the bottom of SM if no seasonality but if there are seasonality, they got pushed down to after seasonal products in buckets 99, 98
  conds = (pdf['BUYER_RATIO_THRESHOLD_BPN'].values == 0) & (pdf['SMART_BASKET_BUCKET'].values == 0) & (pdf['SMART_BASKET_FLAG'].values > 0)
  pdf.loc[conds, "SMART_BASKET_FLAG"] = 0
  # this is equivalent to pdf.loc[conds, "BPN_RANK_FINAL"] += num_sea, but faster
  pdf["BPN_RANK_FINAL"] = num_sea*conds + pdf["BPN_RANK_FINAL"].values
  del conds

  num_keep_sm = (pdf['SMART_BASKET_FLAG'].values > 0).astype(int).sum()
  if num_sea > 0:
    seasonal_derived_columns = [
      'IS_NEAR_FYEAR_14DAYS',
      'HAS_MODEL',
      'AISLE_RANK',
      'SEASONAL_SCORE',
      'SHELF_RANK',
      'SHELF_BPN_RANK',
      'BPN_ID'
    ]
    sort_ascending_logic_seasonal_derived = [False, False, True, False, True, True, True]
    seasonal_columns = seasonal_derived_columns
    sort_ascending_logic_seasonal = sort_ascending_logic_seasonal_derived
    seasonal_rankings = pdf.loc[seasonal_cond, seasonal_columns].sort_values(
      by=seasonal_columns, ascending=sort_ascending_logic_seasonal
    )
    # rankings are always smaller than the sm items' rankings
    pdf.loc[seasonal_rankings.index.values, 'BPN_RANK_FINAL'] = np.arange(seasonal_rankings.shape[0]) + (1 + num_keep_sm)

  #end = time.time()
  #print(f"seasonal_rankings execution time in ms is:  {(end - start)*1000}")

  #==================================================ranking non-seasonal items outside SM or recently purchased seasonal products===========================================#
  # also downrank recently purchased seasonal products
  if do_compute_seasonal:
    seasonal_bucket_conds = np.isin(pdf['SEASONAL_BUCKET'].values, [10, 16, 17])
    # non-seasonal (IS_PAST_SEASONAL_PURCHASE == 0) or past purchased seasonal (from old holiday) (IS_PAST_SEASONAL_PURCHASE == 1)
    conds = (pdf['IS_PAST_SEASONAL_PURCHASE'].values != -1) # IS_PAST_SEASONAL_PURCHASE == -1 meaning it belongs to recent purchase of the incoming holiday
    pdf.loc[conds, 'IS_PAST_SEASONAL_PURCHASE'] = 0
    del conds
    non_sm_cond = ORG_NON_SM_CONDS & (six_month_cond | seasonal_bucket_conds)
  else:
    non_sm_cond = ORG_NON_SM_CONDS & six_month_cond

  bucket_columns = ['BPN_PUR_90', 'BPN_PUR_30']
  sort_ascending_logic_bucket = [False, False]
  #start = time.time()
  if do_compute_seasonal:
    # IS_PAST_SEASONAL_PURCHASE = -1, need to downrank, 0 keep the same rule
    # use SEASONAL_SCORE to break tie, since the frequency columns are likely to be all zeros
    bucket_columns = ['BPN_PUR_90', 'IS_PAST_SEASONAL_PURCHASE', 'BPN_PUR_30', 'SEASONAL_BUCKET', 'SEASONAL_SCORE']
    sort_ascending_logic_bucket = [False, False, False, False, False] 

  frequency2_columns = [
    'TXN_COUNT_BPN', 'PAST_M3_PURCHASE_COUNT_BPN', 'PAST_M6_PURCHASE_COUNT_BPN'
  ]
  sort_ascending_logic_frequency2 = [False, False, False]

  non_sm_columns = bucket_columns + frequency2_columns + derived_columns
  sort_ascending_logic_non_sm = sort_ascending_logic_bucket + \
    sort_ascending_logic_frequency2 + sort_ascending_logic_derived

  if non_sm_cond.astype(int).sum() > 0:
    non_sm_rankings = pdf.loc[non_sm_cond, non_sm_columns].sort_values(
      by=non_sm_columns, ascending=sort_ascending_logic_non_sm
    )
    # rankings are always smaller than the sm items' rankings + highly relevant seasonal products
    pdf.loc[non_sm_rankings.index.values, 'BPN_RANK_FINAL'] = np.arange(non_sm_rankings.shape[0]) + (1 + num_sm + num_sea)
  
  #end = time.time()
  #print(f"non_sm_rankings execution time in ms is:  {(end - start)*1000}")

  #==============================================================================================================================#

  # now set SMART_BASKET_FLAG = 1 for all SMART_BASKET_FLAG > 0 to be consistent
  pdf.loc[(pdf['SMART_BASKET_FLAG'].values > 0), "SMART_BASKET_FLAG"] = 1

def rank_more_6m(pdf, starting_ranking):
  more_six_month_cond = (pdf["RECENCY_BPN"].values >= 180) & (pdf['BPN_RANK_FINAL'].values >= 99999) # some seasonal products were already ranked
  less_thirteen_month_cnt = (more_six_month_cond & (pdf['BPN_RANK_FINAL'].values <= 180000)).astype(int).sum()
  # rank only items that are LESS than six months last run but are now more than 6 months
  new_items_cond = more_six_month_cond & (pdf["BPN_RANK_6_MONTH"].values == 0)
  num_new_items = new_items_cond.astype(int).sum()
  if num_new_items:
    sort_columns = [
      'TXN_COUNT_BPN',
      'AISLE_RANK',
      'SHELF_RANK',
      'SHELF_BPN_RANK',
      'AISLE_BPN_RANK',
      'BPN_ID'
    ]
    sort_ascending_logic = [False, True, True, True, True, True]
    more_6months_rankings = pdf.loc[new_items_cond, sort_columns].sort_values(
        by=sort_columns, ascending=sort_ascending_logic
    )

    # 6 months rankings start at 90,000 (for seasonal products that purchased more than 1 year ago, rankings start at 180,000) for BPN_RANK_6_MONTH
    # note that the BPN_RANK_6_MONTH are not strictly consecutive
    old_items_cond = more_six_month_cond & (pdf["BPN_RANK_6_MONTH"].values > 0)
    if old_items_cond.astype(int).sum():
      pdf["BPN_RANK_6_MONTH"] = old_items_cond*num_new_items + pdf["BPN_RANK_6_MONTH"].values

    pdf.loc[more_6months_rankings.index.values, 'BPN_RANK_6_MONTH'] = \
      np.arange(more_6months_rankings.shape[0]) + (90000+1)

  div = (pdf['BPN_RANK_6_MONTH'].values / 90000).astype(int)
  rem = (pdf['BPN_RANK_6_MONTH'].values % 90000).astype(int)
  adjusment = (div-1)*less_thirteen_month_cnt + (div-1)*rem # adjusment = 0 for rankings ~ 90000

  # if ranking is around 180000, then div = 2, therefore (div*90000+1) removes either 180000 or 90000 in the rankings
  # and adjusment ensures the rankings that are around 180000 are always larger than rankings aroung 90000
  pdf['BPN_RANK_FINAL'] = \
    (1-more_six_month_cond)*pdf['BPN_RANK_FINAL'].values + more_six_month_cond*(pdf['BPN_RANK_6_MONTH'].values - (div*90000+1) + starting_ranking + adjusment)



def bia_algo(feature_df, current_day, active_bpn_df=None, do_seasonal=True, do_lp_ranking=False):

  feature_df = feature_df[feature_df["BATCH_BPN_RANK"].values != -9999999] # remove invalid records still in Cosmo
  
  feature_df['SMART_BASKET_FLAG'] = 0
  feature_df['HOMEPAGE_FLAG'] = 0
  feature_df['BPN_RANK_FINAL'] = 99999

  #start = time.time()
  # calculate time dependent functions  
  get_recency(feature_df, current_day)

  conds_7_13months = (feature_df["RECENCY_BPN"].values < 395) & (feature_df["RECENCY_BPN"].values >= 180)
  conds_more_13months = feature_df["RECENCY_BPN"].values >= 395 # seasonal products

  feature_df['BPN_RANK_6_MONTH'] = (
    conds_more_13months.astype(int)*(180000 + (-1)*feature_df["BATCH_BPN_RANK"].values) +
    conds_7_13months.astype(int)*(90000 + (-1)*feature_df["BATCH_BPN_RANK"].values)
    # 0 if lesss than 6 months
  )
  del conds_7_13months, conds_more_13months

  #end = time.time()
  #print(f"get_recency execution time in ms is:  {(end - start)*1000}")

  #start = time.time()
  get_count_by_month(feature_df)
  #end = time.time()
  #print(f"get_count_by_month execution time in ms is:  {(end - start)*1000}")

  #start = time.time()
  get_relevancy(feature_df) # cython numba does not work in spark
  #end = time.time()
  #print(f"get_relevancy execution time in ms is:  {(end - start)*1000}")

  feature_df["INDEX"] = feature_df["BPN_ID"]
  feature_df.index = feature_df["INDEX"]
  if not do_seasonal:
    do_compute_seasonal = False
  else:
    # IS_SEASONAL = -1000, not seasonal product
    # IS_SEASONAL = 1000, seasonal product but not active
    # IS_SEASONAL != (-1000 and 1000), seasonal product and active
    # IS_PAST_SEASONAL_PURCHASE = 0, not seasonal product so NEITHER
    # IS_PAST_SEASONAL_PURCHASE = 1, seasonal product and were purchased from some old holidays
    # IS_PAST_SEASONAL_PURCHASE = -1, seasonal product and were purchased from the current/incoming holidays
    # SEASONAL_SCORE = 0 for non-seasonal, > 1 for seasonal products with model hence having confidence scores
    # SEASONAL_SCORE = 0 for non-seasonal, < -1 for semi-easonal products with model hence having confidence scores
    # SEASONAL_BUCKET an attribute used in sorting for boosting seasonal products up and down.

    #start = time.time()
    do_compute_seasonal = get_active_seasonal(feature_df, active_bpn_df)
    #end = time.time()
    #print(f"get_active_seasonal execution time in ms is:  {(end - start)*1000}")

  # assign smart basket items
  #start = time.time()
  set_smart_basket_values(feature_df, do_compute_seasonal)
  #end = time.time()
  #print(f"set_smart_basket_values execution time in ms is:  {(end - start)*1000}")
  
  #start = time.time()
  unset_smart_basket_values(feature_df, do_compute_seasonal)
  #end = time.time()
  #print(f"unset_smart_basket_values execution time in ms is:  {(end - start)*1000}")

  # rank items with recency < 6months
  #start = time.time()
  rank_less_6m(feature_df, do_compute_seasonal)
  #end = time.time()
  #print(f"rank_less_6m execution time in ms is:  {(end - start)*1000}")

  num_ranked_items = (feature_df['BPN_RANK_FINAL'].values != 99999).astype(int).sum()
  # if do_lp_ranking == True, then this is for Landing Page, meaning all items
  # if no recent item == 7, then do this because we need to have HOMEPAGE_FLAG for top 35 items
  if do_lp_ranking or num_ranked_items <= 7:
    # rank items with recency > 6months
    #start = time.time()
    rank_more_6m(feature_df, num_ranked_items+1)
    #end = time.time()
    #print(f"rank_more_6m execution time in ms is:  {(end - start)*1000}")
  
  # set homepage flag (top 35 items)
  feature_df.loc[feature_df['BPN_RANK_FINAL'].values <= 35, "HOMEPAGE_FLAG"] = 1
  if num_ranked_items == 0:
    feature_df.loc[feature_df['BPN_RANK_FINAL'].values <= 5, "SMART_BASKET_FLAG"] = 1

  # change rankings to negative for CosmoDB (no need when doing RT)
  #feature_df['BPN_RANK_FINAL'] = (-1) * feature_df['BPN_RANK_FINAL'].values
  
  # when there is a major change in BIA algo, please consider update this value
  # 3 ==> RT and LARGER QSC
  # 4 ==> RT and LARGER QSC and Seasonality
  feature_df['SM_VER'] = 4
  
  sel_cols = [
    "HOUSEHOLD_ID", "BPN_ID", "UPC_ID",
    "DEPT_RANK", "BPN_RANK_FINAL", 
    "SMART_BASKET_FLAG", "HOMEPAGE_FLAG", "SM_VER"
  ]

  #if do_compute_seasonal:
  #  seasonal_cols = ["IS_SEASONAL", "RECENCY_BPN", "SEASONAL_SCORE", "SEASONAL_BUCKET", "IS_PAST_SEASONAL_PURCHASE", "LATEST_TXN_DTE"]
  #  sel_cols += seasonal_cols

  #start = time.time()
  if not do_lp_ranking:
    with_rankings_conds = (feature_df['SMART_BASKET_FLAG'].values == 1) | (feature_df['HOMEPAGE_FLAG'].values == 1)
  else:
    with_rankings_conds = feature_df['BPN_RANK_FINAL'].values < 80000 # remove items that do not have rankings
  
  return_pdf = feature_df.loc[with_rankings_conds, sel_cols].copy()
  #end = time.time()
  #print(f"prepare final df execution time in ms is:  {(end - start)*1000}")
  return return_pdf



def bia_algo_for_pyspark(feature_df, current_day, active_bpn_df=None, do_seasonal=True, do_lp_ranking=False):

  feature_df = feature_df[feature_df["BATCH_BPN_RANK"].values != -9999999] # remove invalid records still in Cosmo
  
  feature_df['SMART_BASKET_FLAG'] = 0
  feature_df['HOMEPAGE_FLAG'] = 0
  feature_df['BPN_RANK_FINAL'] = 99999
  feature_df['SEASONAL_BUCKET'] = 0

  #start = time.time()
  # calculate time dependent functions  
  get_recency(feature_df, current_day)

  conds_7_13months = (feature_df["RECENCY_BPN"].values < 395) & (feature_df["RECENCY_BPN"].values >= 180)
  conds_more_13months = feature_df["RECENCY_BPN"].values >= 395 # seasonal products

  feature_df['BPN_RANK_6_MONTH'] = (
    conds_more_13months.astype(int)*(180000 + (-1)*feature_df["BATCH_BPN_RANK"].values) +
    conds_7_13months.astype(int)*(90000 + (-1)*feature_df["BATCH_BPN_RANK"].values)
    # 0 if lesss than 6 months
  )
  del conds_7_13months, conds_more_13months

  #end = time.time()
  #print(f"get_recency execution time in ms is:  {(end - start)*1000}")

  #start = time.time()
  get_count_by_month(feature_df)
  #end = time.time()
  #print(f"get_count_by_month execution time in ms is:  {(end - start)*1000}")

  #start = time.time()
  get_relevancy_pandas(feature_df) # cython numba can't be serialized in pyspark
  #end = time.time()
  #print(f"get_relevancy execution time in ms is:  {(end - start)*1000}")

  feature_df["INDEX"] = feature_df["BPN_ID"]
  feature_df.index = feature_df["INDEX"]
  if not do_seasonal:
    do_compute_seasonal = False
  else:
    # IS_SEASONAL = -1000, not seasonal product
    # IS_SEASONAL = 1000, seasonal product but not active
    # IS_SEASONAL != (-1000 and 1000), seasonal product and active
    # IS_PAST_SEASONAL_PURCHASE = 0, not seasonal product so NEITHER
    # IS_PAST_SEASONAL_PURCHASE = 1, seasonal product and were purchased from some old holidays
    # IS_PAST_SEASONAL_PURCHASE = -1, seasonal product and were purchased from the current/incoming holidays
    # SEASONAL_SCORE = 0 for non-seasonal, > 1 for seasonal products with model hence having confidence scores
    # SEASONAL_SCORE = 0 for non-seasonal, < -1 for semi-easonal products with model hence having confidence scores
    # SEASONAL_BUCKET an attribute used in sorting for boosting seasonal products up and down.

    #start = time.time()
    do_compute_seasonal = get_active_seasonal(feature_df, active_bpn_df)
    #end = time.time()
    #print(f"get_active_seasonal execution time in ms is:  {(end - start)*1000}")

  # assign smart basket items
  #start = time.time()
  set_smart_basket_values(feature_df, do_compute_seasonal)
  #end = time.time()
  #print(f"set_smart_basket_values execution time in ms is:  {(end - start)*1000}")
  
  #start = time.time()
  unset_smart_basket_values(feature_df, do_compute_seasonal)
  #end = time.time()
  #print(f"unset_smart_basket_values execution time in ms is:  {(end - start)*1000}")

  # rank items with recency < 6months
  #start = time.time()
  rank_less_6m(feature_df, do_compute_seasonal)
  #end = time.time()
  #print(f"rank_less_6m execution time in ms is:  {(end - start)*1000}")

  num_ranked_items = (feature_df['BPN_RANK_FINAL'].values != 99999).astype(int).sum()
  # if do_lp_ranking == True, then this is for Landing Page, meaning all items
  # if no recent item == 7, then do this because we need to have HOMEPAGE_FLAG for top 35 items
  if do_lp_ranking or num_ranked_items <= 7:
    # rank items with recency > 6months
    #start = time.time()
    rank_more_6m(feature_df, num_ranked_items+1)
    #end = time.time()
    #print(f"rank_more_6m execution time in ms is:  {(end - start)*1000}")
  
  # set homepage flag (top 35 items)
  feature_df.loc[feature_df['BPN_RANK_FINAL'].values <= 35, "HOMEPAGE_FLAG"] = 1
  if num_ranked_items == 0:
    feature_df.loc[feature_df['BPN_RANK_FINAL'].values <= 5, "SMART_BASKET_FLAG"] = 1

  # change rankings to negative for CosmoDB (no need when doing RT)
  #feature_df['BPN_RANK_FINAL'] = (-1) * feature_df['BPN_RANK_FINAL'].values
  
  # when there is a major change in BIA algo, please consider update this value
  # 3 ==> RT and LARGER QSC
  # 4 ==> RT and LARGER QSC and Seasonality
  feature_df['SM_VER'] = 4
  
  sel_cols = [
    "HOUSEHOLD_ID", "BPN_ID", "UPC_ID",
    "DEPT_RANK", "BPN_RANK_FINAL", 
    "SMART_BASKET_FLAG", "HOMEPAGE_FLAG", "SM_VER"
  ]

  #if do_compute_seasonal:
  #  seasonal_cols = ["IS_SEASONAL", "RECENCY_BPN", "SEASONAL_SCORE", "SEASONAL_BUCKET", "IS_PAST_SEASONAL_PURCHASE", "LATEST_TXN_DTE"]
  #  sel_cols += seasonal_cols

  #start = time.time()
  if not do_lp_ranking:
    with_rankings_conds = (feature_df['SMART_BASKET_FLAG'].values == 1) | (feature_df['HOMEPAGE_FLAG'].values == 1) |(feature_df['SEASONAL_BUCKET'] > 0)
  else:
    with_rankings_conds = feature_df['BPN_RANK_FINAL'].values < 80000 # remove items that do not have rankings
  
  return_pdf = feature_df.loc[with_rankings_conds, sel_cols].copy()
  #end = time.time()
  #print(f"prepare final df execution time in ms is:  {(end - start)*1000}")
  return return_pdf





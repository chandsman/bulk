from enum import Enum
from typing import Dict

class ModelMapping(Enum):
   
   
    BUY_IT_AGAIN_V0_HOMEPAGE = "getBuyItAgainProducts"
    EGL_FPASS_BANNER_PHASE_1 = "getEglBanners"
    OFFER_BP_PHASE1 = "getOfferBonusPath"
    OFFER_TD_PHASE1 = "getOfferTopDeals"
    OFFER_GRID_SERVICE = "getOfferGridService"
    PRODUCT_THEMES_PHASE1 = "getProductThemes"
    THEMES_PHASE1 = "getThemes"
    BIA_BOOSTED_RECIPES = "getReciperecommendation"
    LATEST_ORDER_BOOSTED_RECIPES = "getReciperecommendation"
    THK_ORDER_2_BANNER_EMAIL_PROMO_CARD = "getRuleBasedBanners"
    THANKSGIVING_2024_CRM_FAV = "getManualCuratedProducts"
    FOODRECIPE_MODEL_TEST = "getManualCuratedRecipes"
    BIA_SEASONAL_HOMEPAGE = "getBuyItAgainProductsRT"
    THK_TURKEY_BANNER_EMAIL_HERO = "getRuleBasedBanners"
    THK_CORE_4_BANNER_EMAIL_PROMO_CARD = "getRuleBasedBanners"
    XMAS_EMAIL_RECS = "getBuyItAgainProducts"
    SEASONAL_PRODUCT_THEMES = "getSeasonalThemes"
    EGL_BANNER_EMAIL_SKINNY = "getRuleBasedBanners"
    BIA_WITH_CATEGORY_FILTER = "getProductfilters"
    BIA_REALTIME_SMARTBASKET = "getBuyItAgainProductsRT"
    BIA_REALTIME_HOMEPAGE = "getBuyItAgainProductsRT"
    CRM_DIVISION_STORE_SHOW_HIDE_BANNER_EMAIL_PROMO_CARD = "getRuleBasedBanners"
    CRM_PERSONA_BUYER_BANNER_EMAIL_HERO = "getRuleBasedBanners"
    RT_RE_RANKING_V1 = "getRtReRankProducts"


    @classmethod
    def get_function_name(cls, model_id: str) -> str:
       
        try:
            return cls[model_id].value
        except KeyError:
            return None

    @classmethod
    def get_all_mappings(cls) -> Dict[str, str]:
        
        return {member.name: member.value for member in cls}
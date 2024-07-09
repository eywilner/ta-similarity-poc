from models.product_matcher import product_match_model
from models.schemas import ProductPairWithAmazon

import logging
from fastapi import FastAPI, HTTPException

# Set up logging to console only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/healthcheck")
def healthcheck():
  return "", 200

@app.post("/match_products")
async def match_products(request_data: ProductPairWithAmazon):

    try:
        logger.info(f"Received request with a product pair\n {request_data}.")
        results = product_match_model(data=request_data)
        return results

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)

from pydantic import BaseModel, HttpUrl, Field

class ProductPairWithAmazon(BaseModel):
    source_title: str
    source_image_url: HttpUrl
    amazon_title: str
    amazon_image_url: HttpUrl
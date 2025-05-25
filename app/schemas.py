import pydantic
from pydantic import BaseModel, Field
class Item(BaseModel):
    BrandName: str = Field(..., 
                           description="Name of the brand")
    Category: str = Field(...,
                           examples=["KidsWear", "WeasternWear", "Footwear"],)
    MRP: float = Field(...,
                       description="Maximum Retail Price of the item")
    Details: str = Field(...,
                       description="Details of the item")
    Sizes: str = Field(...,
                       description="Sizes of the item, can be a string or list of sizes",
                       examples=["S", "M", "L", "XL", "XXL", "S,M,L", "[\"S\", \"M\", \"L\"]"])
  
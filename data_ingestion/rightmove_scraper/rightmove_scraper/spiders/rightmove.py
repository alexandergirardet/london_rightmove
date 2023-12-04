import scrapy


class RightmoveSpider(scrapy.Spider):
    name = "rightmove"
    allowed_domains = ["rightmove.com"]
    start_urls = ["https://rightmove.com"]

    def parse(self, response):
        pass

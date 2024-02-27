# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html
import os

from scrapy import signals
from scrapy import signals
import datetime
from psycopg2.extras import execute_values
from scrapy.signalmanager import dispatcher
import psycopg2

# useful for handling different item types with a single interface
from itemadapter import is_item, ItemAdapter

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

POSTGRES_URI = os.environ.get("MONITORING_URI_PG")


class RightmoveScraperSpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(s.spider_closed, signal=signals.spider_closed)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)

    def spider_closed(self, spider):
        # Retrieve stats
        stats = spider.crawler.stats.get_stats()

        # Call the method to save stats to PostgreSQL
        self.save_stats_to_postgres(stats)

    def save_stats_to_postgres(self, stats):
        # Setup database connection
        logger.info(f"Logging stats to Postgres: {stats}")

        start_time = stats.get("start_time")
        finish_time = stats.get("finish_time")
        elapsed_time_seconds = stats.get("elapsed_time_seconds")
        item_scraped_count = stats.get("item_scraped_count", 0)
        finish_reason = stats.get("finish_reason")
        log_count_debug = stats.get("log_count/DEBUG", 0)
        log_count_info = stats.get("log_count/INFO", 0)
        log_count_error = stats.get("log_count/ERROR", 0)
        mem_usage_startup = stats.get("memusage/startup")
        mem_usage_max = stats.get("memusage/max")
        scheduler_enqueued_memory = stats.get("scheduler/enqueued/memory")
        downloader_request_count = stats.get("downloader/request_count")
        downloader_reponse_count = stats.get("downloader/response_count")
        response_received_count = stats.get("response_received_count")
        downloader_request_method_count_get = stats.get(
            "downloader/request_method_count/GET"
        )
        downloader_request_bytes = stats.get("downloader/request_bytes")

        logger.info("Saving stats to PostgreSQL")
        logger.info(f"start_time: {start_time}")
        logger.info(f"finish_time: {finish_time}")
        logger.info(f"elapsed_time_seconds: {elapsed_time_seconds}")
        logger.info(f"item_scraped_count: {item_scraped_count}")
        logger.info(f"finish_reason: {finish_reason}")
        logger.info(f"log_count_debug: {log_count_debug}")
        logger.info(f"log_count_info: {log_count_info}")
        logger.info(f"log_count_error: {log_count_error}")
        logger.info(f"mem_usage_startup: {mem_usage_startup}")
        logger.info(f"mem_usage_max: {mem_usage_max}")
        logger.info(f"scheduler_enqueued_memory: {scheduler_enqueued_memory}")
        logger.info(f"downloader_request_count: {downloader_request_count}")
        logger.info(f"downloader_reponse_count: {downloader_reponse_count}")
        logger.info(f"response_received_count: {response_received_count}")
        logger.info(
            f"downloader_request_method_count_get: {downloader_request_method_count_get}"
        )
        logger.info(f"downloader_request_bytes: {downloader_request_bytes}")

        insert_sql = """
        INSERT INTO scrapy_rightmove_rental_stats (
            start_time, finish_time, elapsed_time_seconds, item_scraped_count, finish_reason,
            log_count_debug, log_count_info, log_count_error, mem_usage_startup, mem_usage_max, scheduler_enqueued_memory,
            downloader_request_count, downloader_response_count, response_received_count,
            downloader_request_method_count_get, downloader_request_bytes
        ) VALUES %s;
        """

        # Data tuple to insert
        data = (
            stats.get("start_time"),
            stats.get("finish_time"),
            stats.get("elapsed_time_seconds"),
            stats.get("item_scraped_count", 0),
            stats.get("finish_reason"),
            stats.get("log_count/DEBUG", 0),
            stats.get("log_count/INFO", 0),
            stats.get("log_count/ERROR", 0),
            stats.get("memusage/startup"),
            stats.get("memusage/max"),
            stats.get("scheduler/enqueued/memory"),
            stats.get("downloader/request_count"),
            stats.get("downloader/response_count"),
            stats.get("response_received_count"),
            stats.get("downloader/request_method_count/GET"),
            stats.get("downloader/request_bytes"),
        )
        cur = None
        conn = None
        try:
            # Connect to your database
            conn = psycopg2.connect(POSTGRES_URI)
            cur = conn.cursor()

            # Execute the insert statement
            execute_values(cur, insert_sql, [data])

            # Commit the transaction
            conn.commit()

            logger.info("Stats successfully saved to PostgreSQL")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            if cur is not None:
                cur.close()

            if conn is not None:
                conn.close()


class RightmoveScraperDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)

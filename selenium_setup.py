import logging

from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options as ChromeOptions
import os

def create_driver(driver_path, browser, headless=True):
    logging.info(f"Driver object for {browser} will be created from path {driver_path}")
    if 'geckodriver' in driver_path and browser == 'Firefox':
        os.chmod(driver_path, 0o755)
        firefox_options = FirefoxOptions()
        if headless:
            firefox_options.add_argument("-headless")
        webdriver_service = FirefoxService(executable_path=driver_path)
        driver = webdriver.Firefox(service=webdriver_service, options=firefox_options)

    elif browser == 'Chrome':
        chrome_options = ChromeOptions ()
        if headless:
            chrome_options.add_argument ( "--headless" )  # Ensure GUI is off
        chrome_options.add_argument ( "--no-sandbox" )
        chrome_options.add_argument ( "--disable-dev-shm-usage" )

        # Initialize Chrome driver  chromedriver-mac-arm64/chromedriver
        driver = webdriver.Chrome ( service = ChromeService ( ChromeDriverManager (driver_version = '125.0.6422.141' ).install () ),
                                    options = chrome_options )
    return driver
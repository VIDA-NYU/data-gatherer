import logging

from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.chrome.options import Options as ChromeOptions
import os
import time

def create_driver(driver_path, browser, headless=True):
    logging.info(f"Driver object for {browser} will be created from path {driver_path}")
    if 'geckodriver' in driver_path and browser == 'Firefox':
        os.chmod(driver_path, 0o755)
        firefox_options = FirefoxOptions()

        # Enable stealth mode by disabling webdriver flag
        firefox_options.set_preference("dom.webdriver.enabled", False)
        firefox_options.set_preference("useAutomationExtension", False)
        firefox_options.set_preference("general.useragent.override",
                                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/125.0")

        # Prevent headless detection
        if headless:
            firefox_options.add_argument("-headless")
        else:
            firefox_options.set_preference("dom.webnotifications.enabled", False)  # Disable pop-ups
            firefox_options.set_preference("dom.push.enabled", False)

        # Disable automation flags
        firefox_options.set_preference("marionette.logging", False)
        firefox_options.set_preference("devtools.jsonview.enabled", False)
        firefox_options.set_preference("privacy.resistFingerprinting", False)  # Prevent canvas fingerprinting
        firefox_options.set_preference("webdriver.log.driver", "OFF")  # Silence logs

        # Set a valid window size (prevents detection)
        firefox_options.add_argument("--width=1920")
        firefox_options.add_argument("--height=1080")

        logging.info(f"Driver options {firefox_options}")

        # Initialize WebDriver
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
    else:
        raise ValueError(f"Unsupported browser {browser}")

    return driver
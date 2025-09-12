import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.chrome import ChromeDriverManager

def create_driver(driver_path=None, browser="Firefox", headless=True, logger=None, download_dir="output/suppl_files"):
    logger.info(f"Creating WebDriver for browser: {browser}, with driver located at driver_path: {driver_path}")

    if browser.lower() == 'firefox':
        firefox_options = FirefoxOptions()

        if headless:
            firefox_options.add_argument("-headless")

        # Set preferences directly in FirefoxOptions (not using FirefoxProfile)
        firefox_options.set_preference("browser.download.folderList", 2)
        firefox_options.set_preference("browser.download.dir", os.path.abspath(download_dir))
        firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk",
            "application/pdf,application/octet-stream,application/zip,"
            "application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        firefox_options.set_preference("pdfjs.disabled", True)
        firefox_options.set_preference("browser.download.manager.showWhenStarting", False)

        # Additional stealth preferences
        firefox_options.set_preference("dom.webdriver.enabled", False)
        firefox_options.set_preference("useAutomationExtension", False)
        firefox_options.set_preference("general.useragent.override",
                                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/125.0")
        firefox_options.add_argument("--width=1920")
        firefox_options.add_argument("--height=1080")

        if driver_path:
            os.chmod(driver_path, 0o755)
            service = FirefoxService(executable_path=driver_path, log_path="logs/geckodriver.log")
            logger.info(f"Using provided Firefox driver path: {driver_path}")
        else:
            logger.info("No driver path provided, using GeckoDriverManager to auto-install Firefox driver.")
            service = FirefoxService(executable_path=GeckoDriverManager().install(), log_path="logs/geckodriver.log")
            logger.info(f"Using GeckoDriverManager to auto-install Firefox driver {service}.") if logger else None

        driver = webdriver.Firefox(service=service, options=firefox_options)
        logger.info("Firefox WebDriver created successfully.")

    elif browser.lower() == 'chrome':
        chrome_options = ChromeOptions()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        if driver_path:
            service = ChromeService(executable_path=driver_path)
        else:
            logger.info("No driver path provided, using ChromeDriverManager to auto-install Chrome driver.")
            service = ChromeService(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=chrome_options)
        logger.info("Chrome WebDriver created successfully.")

    else:
        raise ValueError(f"Unsupported browser: {browser}")

    return driver
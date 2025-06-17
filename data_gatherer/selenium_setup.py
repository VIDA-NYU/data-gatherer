import os
import hashlib
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.chrome import ChromeDriverManager

def verify_driver_checksum(driver_path, expected_sha256, logger=None):
    """Verify the SHA-256 checksum of the downloaded driver binary."""
    sha256 = hashlib.sha256()
    with open(driver_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if logger:
        logger.info(f"Verifying driver at {driver_path}: expected {expected_sha256}, actual {actual}")
    if actual != expected_sha256:
        raise RuntimeError(f"Driver checksum mismatch: expected {expected_sha256}, got {actual}")

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
            driver_path = GeckoDriverManager().install()
            # TODO: Replace with the actual expected SHA-256 for the downloaded version
            expected_sha256 = os.environ.get("GECKODRIVER_SHA256", "")
            if expected_sha256:
                verify_driver_checksum(driver_path, expected_sha256, logger)
            else:
                logger.warning("No expected SHA-256 provided for geckodriver; skipping checksum verification.")
            service = FirefoxService(executable_path=driver_path, log_path="logs/geckodriver.log")
            logger.info(f"Using GeckoDriverManager to auto-install Firefox driver {service}.") if logger else None

        driver = webdriver.Firefox(service=service, options=firefox_options)

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
            driver_path = ChromeDriverManager().install()
            # TODO: Replace with the actual expected SHA-256 for the downloaded version
            expected_sha256 = os.environ.get("CHROMEDRIVER_SHA256", "")
            if expected_sha256:
                verify_driver_checksum(driver_path, expected_sha256, logger)
            else:
                logger.warning("No expected SHA-256 provided for chromedriver; skipping checksum verification.")
            service = ChromeService(driver_path)

        driver = webdriver.Chrome(service=service, options=chrome_options)

    else:
        raise ValueError(f"Unsupported browser: {browser}")

    return driver

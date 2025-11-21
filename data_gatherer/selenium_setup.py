import os
import platform
import subprocess
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.chrome import ChromeDriverManager

def create_driver(driver_path=None, browser="Firefox", headless=True, logger=None, download_dir="scripts/downloads"):
    logger.info(f"Creating WebDriver for browser: {browser}, with driver located at driver_path: {driver_path}")

    if browser is None or browser.lower() == 'firefox':
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
            
            # Handle Apple Silicon Macs - force correct architecture
            import platform
            if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                logger.info("Detected Apple Silicon Mac (ARM64) - ensuring compatible geckodriver")
                # Delete old incompatible geckodriver if exists
                old_mac64_path = os.path.expanduser("~/.wdm/drivers/geckodriver/mac64")
                if os.path.exists(old_mac64_path):
                    logger.info(f"Removing old Intel geckodriver from: {old_mac64_path}")
                    try:
                        import shutil
                        shutil.rmtree(old_mac64_path)
                    except Exception as e:
                        logger.warning(f"Could not remove old geckodriver: {e}")
            
            geckodriver_path = GeckoDriverManager().install()
            logger.info(f"Geckodriver path: {geckodriver_path}")
            
            # Set execute permissions and remove quarantine attributes
            try:
                os.chmod(geckodriver_path, 0o755)
                logger.info(f"Set execute permissions (755) on geckodriver")
                
                # Remove macOS quarantine/provenance attributes
                if platform.system() == 'Darwin':
                    try:
                        # Remove all extended attributes
                        subprocess.run(['xattr', '-c', geckodriver_path], 
                                      capture_output=True, check=False)
                        logger.info(f"Removed macOS extended attributes from geckodriver")
                    except Exception as e:
                        logger.warning(f"Could not remove extended attributes: {e}")
            except Exception as e:
                logger.warning(f"Error setting geckodriver permissions: {e}")
            
            service = FirefoxService(executable_path=geckodriver_path, log_path="logs/geckodriver.log")
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
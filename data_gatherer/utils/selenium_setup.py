from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import os

def create_driver(driver_path=None, browser_name='Firefox', headless=True):
    """Create and configure a WebDriver instance based on specified parameters.

    Args:
        driver_path (str, optional): Path to browser driver executable.
        browser_name (str, optional): Browser to use ('Firefox', 'Chrome', 'Edge').
        headless (bool, optional): Run in headless mode. Defaults to True.

    Returns:
        WebDriver: Configured WebDriver instance.
    """
    if browser_name.lower() == 'firefox':
        options = FirefoxOptions()
        if headless:
            options.add_argument('--headless')
        
        # Use WebDriver Manager if no path provided
        if not driver_path:
            from webdriver_manager.firefox import GeckoDriverManager
            driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(), options=options)
        else:
            driver = webdriver.Firefox(executable_path=driver_path, options=options)
    
    elif browser_name.lower() == 'chrome':
        options = ChromeOptions()
        if headless:
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')  # Required for headless on Windows
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Use WebDriver Manager if no path provided
        if not driver_path:
            driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        else:
            driver = webdriver.Chrome(executable_path=driver_path, options=options)
    
    elif browser_name.lower() == 'edge':
        options = EdgeOptions()
        if headless:
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
        
        # Use WebDriver Manager if no path provided
        if not driver_path:
            driver = webdriver.Edge(EdgeChromiumDriverManager().install(), options=options)
        else:
            driver = webdriver.Edge(executable_path=driver_path, options=options)
    
    else:
        raise ValueError(f"Unsupported browser: {browser_name}. Use 'Firefox', 'Chrome', or 'Edge'.")
    
    # Set up driver behavior
    driver.set_page_load_timeout(60)  # 60 seconds timeout for page loads
    driver.implicitly_wait(10)  # Wait up to 10 seconds for elements to become available
    
    return driver

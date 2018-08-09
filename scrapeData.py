from selenium import webdriver
from selenium.webdriver import ActionChains
import requests

driver = webdriver.Firefox()
# Using bing because the image search is more accurate than googles image search
#driver.get("https://www.bing.com/images/search?q=b2+stealth+bomber&FORM=HDRSC2")
#driver.get("https://www.bing.com/images/search?q=b1-b&FORM=HDRSC2")
driver.get("https://www.bing.com/images/search?q=sr-71&qs=n&form=QBIR&sp=-1&pq=sr-71&sc=8-5&sk=&cvid=CB9A80D0D2574F89AD1A049B5937924A")
images = driver.find_elements_by_class_name("mimg")
actions = ActionChains(driver) 
counter = 0
for image in images: # get all images that are on the page
	#actions.context_click(image).perform()
	src = image.get_attribute("src")
	print(src)
	try:
		r = requests.get(src, stream=True)
		if r.status_code == 200:
    			with open("C:/Users/markw/Pictures/airplanes/SR-71/sr-71-" + str(counter) + ".png", 'wb') as f:
        			for chunk in r:
            				f.write(chunk)
	except:
		pass
	counter += 1
print(images)

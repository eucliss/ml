import requests
from bs4 import BeautifulSoup
import base64
#import database from local ignorant lib folder
from course_database import GolfDB
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class ProvisData():
    def __init__(self) -> None:
        self.data = None
        self.base_url = "https://www.provisualizer.com/"
        self.courses = []
        self.db = GolfDB()
        self.aronimink = {'course': 'aronimink', 'course_id': '974', '2d': '/2dlink.php?id=974', '3d': '/3dlink.php?id=974', 'mobile': '/mobilelink.php?id=974', 'yardage': '/yardagelink.php?id=974'}

    def get_webpage(self, url:str):
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.text
            # parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            # print(soup)
            # continue with the rest of your code
            #get all hrefs
            return soup
        else:
            print("issue getting soup")
            print(response)
            return None

    def gather_list_of_courses(self):
        # get the webpage with the list of courses
        url = f'{self.base_url}/courses/'  # replace with the actual URL
        soup = self.get_webpage(url)
        if soup is not None:
            courses = []
            for link in soup.find_all('a'):
                course_link = link.get('href')
                if '=' in course_link or len(course_link) < 2:
                    continue
                else:
                    courses.append(course_link.split('.')[0])
            self.courses = courses
    
    def get_course_id(self, course:str):    
        # get the landing page data
        url = f'{self.base_url}/courses/{course}.php'
        soup = self.get_webpage(url)
        if soup is None:
            print("soup none get course data")
        planner_links = soup.find_all('td', class_='plannerlink')
        plan_links = []
        for link in planner_links:
            plan_links.append(link.find('a').get('href'))
        course_id = plan_links[0].split('=')[-1]
        return course_id

    def get_course_data(self, course:str):    
        # get the landing page data
        url = f'{self.base_url}/courses/{course}.php'
        soup = self.get_webpage(url)
        if soup is None:
            print("soup none get course data")
        planner_links = soup.find_all('td', class_='plannerlink')
        plan_links = {}
        for link in planner_links:
            extension = link.find('a').get('href')
            if '2dlink' in extension:
                plan_links['2d'] = extension
            elif '3dlink' in extension:
                plan_links['3d'] = extension
            elif 'mobile' in extension:
                plan_links['mobile'] = extension
            elif 'yardage' in extension:
                plan_links['yardage'] = extension
            else:
                continue
                
        course_id = plan_links['2d'].split('=')[-1]
        return {
            'course': course,
            'course_id':course_id,
            '2d':plan_links['2d'],
            '3d':plan_links['3d'],
            'mobile':plan_links['mobile'],
            'yardage':plan_links['yardage']
        }
    
    def populate_course_id_db(self):
        if len(self.courses) == 0:
            self.gather_list_of_courses()
        for course in self.courses:
            try:
                print(f"Getting course data for {course}")
                course_data = self.get_course_data(course)
                print(f"Storing course data for {course_data}")
                self.db.storeCourseID(course_data)
            except Exception as e:
                print(f"Error getting course data for {course}: {e}")
    
   # use selenium to get a screenshot of the website
    def screenshot_course(self, course:str):
        # get the webpage with the list of courses
        url = f'{self.base_url}/{self.aronimink["yardage"]}'  # replace with the actual URL
        driver = webdriver.Chrome()
        #wait for the page to load
        driver.get(url)

        # Find the canvas element
        canvas = driver.find_element(By.CSS_SELECTOR, "canvas")
        driver.find_element(By.CSS_SELECTOR, '.button').click()

        # Get the data URL of the canvas element using JavaScript
        canvas_data_url = driver.execute_script("return arguments[0].toDataURL('image/jpeg');", canvas)

        # Strip the meta information (i.e., "data:image/jpeg;base64,") from the data URL
        base64_encoded_data = canvas_data_url.split(',')[1]

        # Decode the base64 string into binary data
        image_data = base64.b64decode(base64_encoded_data)

        # Write the binary data to an image file
        with open("canvas_image.jpg", "wb") as file:
            file.write(image_data)
        # driver.implicitly_wait(100)
        # driver.save_screenshot(f'{course}.png')
        # driver.quit()
        


p = ProvisData()
# p.db.kill
# p.db.kill()
# print(p.db.getCollectionCount('course_ids'))
# p.db.kill()
# p.populate_course_id_db()
# print(p.get_course_data('aronimink'))
p.screenshot_course('aronimink')

#huntingdale
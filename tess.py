import os
import pandas as pd
import numpy as py
import cv2
import numpy as np
from PIL import Image
import pytesseract
import argparse
import re
import spacy
from datetime import datetime
from nltk.tokenize import SpaceTokenizer
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import conll2000
from nltk.chunk import ne_chunk
from nltk import pos_tag



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])


#preprocessing

scale_percent = 60 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

alpha = 1.1 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(resized, alpha=alpha, beta=beta)


gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Image", gray)

# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)



# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
filename = open("text.txt","w") 
filename.write(text)
filename.close()

filename = ("text.txt")
with open(filename,"r", encoding="utf-8") as file:
	filedata = file.readlines()
data_cleaned = ""
for line_data in filedata:
	data_cleaned += line_data.replace('\n',' ')

def getdate(data_cleaned):
    actual_date = []
    re1 = r'[0-31]{1,2}/[1-12]{1,2}/[\d]4}'
    re2 = r'[0-31]{1,2}-[1-12]{1,2}-[\d]4}'
    re3 = r'[0-31]{1,2}.[1-12]{1,2}.[\d]{4}'
    re4 = r'[0-31]{1,2} [ADFJMNOS]\w* [\d]{4}'
    re5 = r'Date:[0-31]{1,2}-[ADFJMNOS]\w*-[\d]{4}'
    re6 = r'[0-31]{1,2}/[ADFJMNOS]\w*/[\d]{4}'
    re7 = r'Date:[0-31]{1,2}/[ADFJMNOS]\w*/[\d]{4}' 
    re8 = r'[0-31]{1,2}-[ADFJMNOS]\w*-[\d]{4}'
    re9 = r'[0-31]{1,2}/[1-12]{1,2}/[0-20]{2}'
    re10 = r'[0-31]{1,2}-[1-12]{1,2}-[0-20]{2}'
    re11 = r'[0-31]{1,2}.[1-12]{1,2}.[0-20]{2}'
    re12 = r'[0-31]{1,2} [ADFJMNOS]\w* [0-20]{2}'
    re13 = r'Date:[0-31]{1,2}-[ADFJMNOS]\w*-[0-20]{2}' 
    re14 = r'[0-31]{1,2}/[ADFJMNOS]\w*/[0-20]{2}'
    re15 = r'Date:[0-31]{1,2}/[ADFJMNOS]\w*/[0-20]{2}'
    re16 = r'[0-31]{1,2}-[ADFJMNOS]\w*-[0-20]{2}'



    generic_re = re.compile("(%s|%s|%s|%s|%s|%s|%s|%s)" % (re1,re2,re3,re4,re5,re6,re7,re8)).findall(data_cleaned)
    if len(generic_re)>1:
       bill_date=generic_re[1]
    else:
       bill_date=generic_re[0]
    return bill_date
invoice_date=getdate(data_cleaned)
print(invoice_date)

test=['Mumbai', 'Delhi', 'Bengaluru', 'Ahmedabad', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Jaipur', 'Surat', 'Lucknow', 'Kanpur', 'Nagpur', 'Patna', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Vadodara', 'Firozabad', 'Ludhiana', 'Rajkot', 'Agra', 'Siliguri', 'Nashik', 'Faridabad', 'Patiala', 'Meerut', 'Kalyan-Dombivali', 'Vasai-Virar', 'Varanasi', 'Srinagar', 'Dhanbad', 'Jodhpur', 'Amritsar', 'Raipur', 'Allahabad', 'Coimbatore', 'Jabalpur', 'Gwalior', 'Vijayawada', 'Madurai', 'Guwahati', 'Chandigarh', 'Hubli-Dharwad', 'Amroha', 'Moradabad', 'Gurgaon', 'Aligarh', 'Solapur', 'Ranchi', 'Jalandhar', 'Tiruchirappalli', 'Bhubaneswar', 'Salem', 'Warangal', 'Mira-Bhayandar', 'Thiruvananthapuram', 'Bhiwandi', 'Saharanpur', 'Guntur', 'Amravati', 'Bikaner', 'Noida', 'Jamshedpur', 'Bhilai Nagar', 'Cuttack', 'Kochi', 'Udaipur', 'Bhavnagar', 'Dehradun', 'Asansol', 'Nanded-Waghala', 'Ajmer', 'Jamnagar', 'Ujjain', 'Sangli', 'Loni', 'Jhansi', 'Pondicherry', 'Nellore', 'Jammu', 'Belagavi', 'Raurkela', 'Mangaluru', 'Tirunelveli', 'Malegaon', 'Gaya', 'Tiruppur', 'Davanagere', 'Kozhikode', 'Akola', 'Kurnool', 'Bokaro Steel City', 'Rajahmundry', 'Ballari', 'Agartala', 'Bhagalpur', 'Latur', 'Dhule', 'Korba', 'Bhilwara', 'Brahmapur', 'Mysore', 'Muzaffarpur', 'Ahmednagar', 'Kollam', 'Raghunathganj', 'Bilaspur', 'Shahjahanpur', 'Thrissur', 'Alwar', 'Kakinada', 'Nizamabad', 'Sagar', 'Tumkur', 'Hisar', 'Rohtak', 'Panipat', 'Darbhanga', 'Kharagpur', 'Aizawl', 'Ichalkaranji', 'Tirupati', 'Karnal', 'Bathinda', 'Rampur', 'Shivamogga', 'Ratlam', 'Modinagar', 'Durg', 'Shillong', 'Imphal', 'Hapur', 'Ranipet', 'Anantapur', 'Arrah', 'Karimnagar', 'Parbhani', 'Etawah', 'Bharatpur', 'Begusarai', 'New Delhi', 'Chhapra', 'Kadapa', 'Ramagundam', 'Pali', 'Satna', 'Vizianagaram', 'Katihar', 'Hardwar', 'Sonipat', 'Nagercoil', 'Thanjavur', 'Murwara (Katni)', 'Naihati', 'Sambhal', 'Nadiad', 'Yamunanagar', 'English Bazar', 'Eluru', 'Munger', 'Panchkula', 'Raayachuru', 'Panvel', 'Deoghar', 'Ongole', 'Nandyal', 'Morena', 'Bhiwani', 'Porbandar', 'Palakkad', 'Anand', 'Purnia', 'Baharampur', 'Barmer', 'Morvi', 'Orai', 'Bahraich', 'Sikar', 'Vellore', 'Singrauli', 'Khammam', 'Mahesana', 'Silchar', 'Sambalpur', 'Rewa', 'Unnao', 'Hugli-Chinsurah', 'Raiganj', 'Phusro', 'Adityapur', 'Alappuzha', 'Bahadurgarh', 'Machilipatnam', 'Rae Bareli', 'Jalpaiguri', 'Bharuch', 'Pathankot', 'Hoshiarpur', 'Baramula', 'Adoni', 'Jind', 'Tonk', 'Tenali', 'Kancheepuram', 'Vapi', 'Sirsa', 'Navsari', 'Mahbubnagar', 'Puri', 'Robertson Pet', 'Erode', 'Batala', 'Haldwani-cum-Kathgodam', 'Vidisha', 'Saharsa', 'Thanesar', 'Chittoor', 'Veraval', 'Lakhimpur', 'Sitapur', 'Hindupur', 'Santipur', 'Balurghat', 'Ganjbasoda', 'Moga', 'Proddatur', 'Srinagar', 'Medinipur', 'Habra', 'Sasaram', 'Hajipur', 'Bhuj', 'Shivpuri', 'Ranaghat', 'Shimla', 'Tiruvannamalai', 'Kaithal', 'Rajnandgaon', 'Godhra', 'Hazaribag', 'Bhimavaram', 'Mandsaur', 'Dibrugarh', 'Kolar', 'Bankura', 'Mandya', 'Dehri-on-Sone', 'Madanapalle', 'Malerkotla', 'Lalitpur', 'Bettiah', 'Pollachi', 'Khanna', 'Neemuch', 'Palwal', 'Palanpur', 'Guntakal', 'Nabadwip', 'Udupi', 'Jagdalpur', 'Motihari', 'Pilibhit', 'Dimapur', 'Mohali', 'Sadulpur', 'Rajapalayam', 'Dharmavaram', 'Kashipur', 'Sivakasi', 'Darjiling', 'Chikkamagaluru', 'Gudivada', 'Baleshwar Town', 'Mancherial', 'Srikakulam', 'Adilabad', 'Yavatmal', 'Barnala', 'Nagaon', 'Narasaraopet', 'Raigarh', 'Roorkee', 'Valsad', 'Ambikapur', 'Giridih', 'Chandausi', 'Purulia', 'Patan', 'Bagaha', 'Hardoi ', 'Achalpur', 'Osmanabad', 'Deesa', 'Nandurbar', 'Azamgarh', 'Ramgarh', 'Firozpur', 'Baripada Town', 'Karwar', 'Siwan', 'Rajampet', 'Pudukkottai', 'Anantnag', 'Tadpatri', 'Satara', 'Bhadrak', 'Kishanganj', 'Suryapet', 'Wardha', 'Ranebennuru', 'Amreli', 'Neyveli (TS)', 'Jamalpur', 'Marmagao', 'Udgir', 'Tadepalligudem', 'Nagapattinam', 'Buxar', 'Aurangabad', 'Jehanabad', 'Phagwara', 'Khair', 'Sawai Madhopur', 'Kapurthala', 'Chilakaluripet', 'Aurangabad', 'Malappuram', 'Rewari', 'Nagaur', 'Sultanpur', 'Nagda', 'Port Blair', 'Lakhisarai', 'Panaji', 'Tinsukia', 'Itarsi', 'Kohima', 'Balangir', 'Nawada', 'Jharsuguda', 'Jagtial', 'Viluppuram', 'Amalner', 'Zirakpur', 'Tanda', 'Tiruchengode', 'Nagina', 'Yemmiganur', 'Vaniyambadi', 'Sarni', 'Theni Allinagaram', 'Margao', 'Akot', 'Sehore', 'Mhow Cantonment', 'Kot Kapura', 'Makrana', 'Pandharpur', 'Miryalaguda', 'Shamli', 'Seoni', 'Ranibennur', 'Kadiri', 'Shrirampur', 'Rudrapur', 'Parli', 'Najibabad', 'Nirmal', 'Udhagamandalam', 'Shikohabad', 'Jhumri Tilaiya', 'Aruppukkottai', 'Ponnani', 'Jamui', 'Sitamarhi', 'Chirala', 'Anjar', 'Karaikal', 'Hansi', 'Anakapalle', 'Mahasamund', 'Faridkot', 'Saunda', 'Dhoraji', 'Paramakudi', 'Balaghat', 'Sujangarh', 'Khambhat', 'Muktsar', 'Rajpura', 'Kavali', 'Dhamtari', 'Ashok Nagar', 'Sardarshahar', 'Mahuva', 'Bargarh', 'Kamareddy', 'Sahibganj', 'Kothagudem', 'Ramanagaram', 'Gokak', 'Tikamgarh', 'Araria', 'Rishikesh', 'Shahdol', 'Medininagar (Daltonganj)', 'Arakkonam', 'Washim', 'Sangrur', 'Bodhan', 'Fazilka', 'Palacole', 'Keshod', 'Sullurpeta', 'Wadhwan', 'Gurdaspur', 'Vatakara', 'Tura', 'Narnaul', 'Kharar', 'Yadgir', 'Ambejogai', 'Ankleshwar', 'Savarkundla', 'Paradip', 'Virudhachalam', 'Kanhangad', 'Kadi', 'Srivilliputhur', 'Gobindgarh', 'Tindivanam', 'Mansa', 'Taliparamba', 'Manmad', 'Tanuku', 'Rayachoti', 'Virudhunagar', 'Koyilandy', 'Jorhat', 'Karur', 'Valparai', 'Srikalahasti', 'Neyyattinkara', 'Bapatla', 'Fatehabad', 'Malout', 'Sankarankovil', 'Tenkasi', 'Ratnagiri', 'Rabkavi Banhatti', 'Sikandrabad', 'Chaibasa', 'Chirmiri', 'Palwancha', 'Bhawanipatna', 'Kayamkulam', 'Pithampur', 'Nabha', 'Shahabad, Hardoi', 'Dhenkanal', 'Uran Islampur', 'Gopalganj', 'Bongaigaon City', 'Palani', 'Pusad', 'Sopore', 'Pilkhuwa', 'Tarn Taran', 'Renukoot', 'Mandamarri', 'Shahabad', 'Barbil', 'Koratla', 'Madhubani', 'Arambagh', 'Gohana', 'Ladnu', 'Pattukkottai', 'Sirsi', 'Sircilla', 'Tamluk', 'Jagraon', 'AlipurdUrban Agglomerationr', 'Alirajpur', 'Tandur', 'Naidupet', 'Tirupathur', 'Tohana', 'Ratangarh', 'Dhubri', 'Masaurhi', 'Visnagar', 'Vrindavan', 'Nokha', 'Nagari', 'Narwana', 'Ramanathapuram', 'Ujhani', 'Samastipur', 'Laharpur', 'Sangamner', 'Nimbahera', 'Siddipet', 'Suri', 'Diphu', 'Jhargram', 'Shirpur-Warwade', 'Tilhar', 'Sindhnur', 'Udumalaipettai', 'Malkapur', 'Wanaparthy', 'Gudur', 'Kendujhar', 'Mandla', 'Mandi', 'Nedumangad', 'North Lakhimpur', 'Vinukonda', 'Tiptur', 'Gobichettipalayam', 'Sunabeda', 'Wani', 'Upleta', 'Narasapuram', 'Nuzvid', 'Tezpur', 'Una', 'Markapur', 'Sheopur', 'Thiruvarur', 'Sidhpur', 'Sahaswan', 'Suratgarh', 'Shajapur', 'Rayagada', 'Lonavla', 'Ponnur', 'Kagaznagar', 'Gadwal', 'Bhatapara', 'Kandukur', 'Sangareddy', 'Unjha', 'Lunglei', 'Karimganj', 'Kannur', 'Bobbili', 'Mokameh', 'Talegaon Dabhade', 'Anjangaon', 'Mangrol', 'Sunam', 'Gangarampur', 'Thiruvallur', 'Tirur', 'Rath', 'Jatani', 'Viramgam', 'Rajsamand', 'Yanam', 'Kottayam', 'Panruti', 'Dhuri', 'Namakkal', 'Kasaragod', 'Modasa', 'Rayadurg', 'Supaul', 'Kunnamkulam', 'Umred', 'Bellampalle', 'Sibsagar', 'Mandi Dabwali', 'Ottappalam', 'Dumraon', 'Samalkot', 'Jaggaiahpet', 'Goalpara', 'Tuni', 'Lachhmangarh', 'Bhongir', 'Amalapuram', 'Firozpur Cantt.', 'Vikarabad', 'Thiruvalla', 'Sherkot', 'Palghar', 'Shegaon', 'Jangaon', 'Bheemunipatnam', 'Panna', 'Thodupuzha', 'KathUrban Agglomeration', 'Palitana', 'Arwal', 'Venkatagiri', 'Kalpi', 'Rajgarh (Churu)', 'Sattenapalle', 'Arsikere', 'Ozar', 'Thirumangalam', 'Petlad', 'Nasirabad', 'Phaltan', 'Rampurhat', 'Nanjangud', 'Forbesganj', 'Tundla', 'BhabUrban Agglomeration', 'Sagara', 'Pithapuram', 'Sira', 'Bhadrachalam', 'Charkhi Dadri', 'Chatra', 'Palasa Kasibugga', 'Nohar', 'Yevla', 'Sirhind Fatehgarh Sahib', 'Bhainsa', 'Parvathipuram', 'Shahade', 'Chalakudy', 'Narkatiaganj', 'Kapadvanj', 'Macherla', 'Raghogarh-Vijaypur', 'Rupnagar', 'Naugachhia', 'Sendhwa', 'Byasanagar', 'Sandila', 'Gooty', 'Salur', 'Nanpara', 'Sardhana', 'Vita', 'Gumia', 'Puttur', 'Jalandhar Cantt.', 'Nehtaur', 'Changanassery', 'Mandapeta', 'Dumka', 'Seohara', 'Umarkhed', 'Madhupur', 'Vikramasingapuram', 'Punalur', 'Kendrapara', 'Sihor', 'Nellikuppam', 'Samana', 'Warora', 'Nilambur', 'Rasipuram', 'Ramnagar', 'Jammalamadugu', 'Nawanshahr', 'Thoubal', 'Athni', 'Cherthala', 'Sidhi', 'Farooqnagar', 'Peddapuram', 'Chirkunda', 'Pachora', 'Madhepura', 'Pithoragarh', 'Tumsar', 'Phalodi', 'Tiruttani', 'Rampura Phul', 'Perinthalmanna', 'Padrauna', 'Pipariya', 'Dalli-Rajhara', 'Punganur', 'Mattannur', 'Mathura', 'Thakurdwara', 'Nandivaram-Guduvancheri', 'Mulbagal', 'Manjlegaon', 'Wankaner', 'Sillod', 'Nidadavole', 'Surapura', 'Rajagangapur', 'Sheikhpura', 'Parlakhemundi', 'Kalimpong', 'Siruguppa', 'Arvi', 'Limbdi', 'Barpeta', 'Manglaur', 'Repalle', 'Mudhol', 'Shujalpur', 'Mandvi', 'Thangadh', 'Sironj', 'Nandura', 'Shoranur', 'Nathdwara', 'Periyakulam', 'Sultanganj', 'Medak', 'Narayanpet', 'Raxaul Bazar', 'Rajauri', 'Pernampattu', 'Nainital', 'Ramachandrapuram', 'Vaijapur', 'Nangal', 'Sidlaghatta', 'Punch', 'Pandhurna', 'Wadgaon Road', 'Talcher', 'Varkala', 'Pilani', 'Nowgong', 'Naila Janjgir', 'Mapusa', 'Vellakoil', 'Merta City', 'Sivaganga', 'Mandideep', 'Sailu', 'Vyara', 'Kovvur', 'Vadalur', 'Nawabganj', 'Padra', 'Sainthia', 'Siana', 'Shahpur', 'Sojat', 'Noorpur', 'Paravoor', 'Murtijapur', 'Ramnagar', 'Sundargarh', 'Taki', 'Saundatti-Yellamma', 'Pathanamthitta', 'Wadi', 'Rameshwaram', 'Tasgaon', 'Sikandra Rao', 'Sihora', 'Tiruvethipuram', 'Tiruvuru', 'Mehkar', 'Peringathur', 'Perambalur', 'Manvi', 'Zunheboto', 'Mahnar Bazar', 'Attingal', 'Shahbad', 'Puranpur', 'Nelamangala', 'Nakodar', 'Lunawada', 'Murshidabad', 'Mahe', 'Lanka', 'Rudauli', 'Tuensang', 'Lakshmeshwar', 'Zira', 'Yawal', 'Thana Bhawan', 'Ramdurg', 'Pulgaon', 'Sadasivpet', 'Nargund', 'Neem-Ka-Thana', 'Memari', 'Nilanga', 'Naharlagun', 'Pakaur', 'Wai', 'Tarikere', 'Malavalli', 'Raisen', 'Lahar', 'Uravakonda', 'Savanur', 'Sirohi', 'Udhampur', 'Umarga', 'Pratapgarh', 'Lingsugur', 'Usilampatti', 'Palia Kalan', 'Wokha', 'Rajpipla', 'Vijayapura', 'Rawatbhata', 'Sangaria', 'Paithan', 'Rahuri', 'Patti', 'Zaidpur', 'Lalsot', 'Maihar', 'Vedaranyam', 'Nawapur', 'Solan', 'Vapi', 'Sanawad', 'Warisaliganj', 'Revelganj', 'Sabalgarh', 'Tuljapur', 'Simdega', 'Musabani', 'Kodungallur', 'Phulabani', 'Umreth', 'Narsipatnam', 'Nautanwa', 'Rajgir', 'Yellandu', 'Sathyamangalam', 'Pilibanga', 'Morshi', 'Pehowa', 'Sonepur', 'Pappinisseri', 'Zamania', 'Mihijam', 'Purna', 'Puliyankudi', 'Shikarpur, Bulandshahr', 'Umaria', 'Porsa', 'Naugawan Sadat', 'Fatehpur Sikri', 'Manuguru', 'Udaipur', 'Pipar City', 'Pattamundai', 'Nanjikottai', 'Taranagar', 'Yerraguntla', 'Satana', 'Sherghati', 'Sankeshwara', 'Madikeri', 'Thuraiyur', 'Sanand', 'Rajula', 'Kyathampalle', 'Shahabad, Rampur', 'Tilda Newra', 'Narsinghgarh', 'Chittur-Thathamangalam', 'Malaj Khand', 'Sarangpur', 'Robertsganj', 'Sirkali', 'Radhanpur', 'Tiruchendur', 'Utraula', 'Patratu', 'Vijainagar, Ajmer', 'Periyasemur', 'Pathri', 'Sadabad', 'Talikota', 'Sinnar', 'Mungeli', 'Sedam', 'Shikaripur', 'Sumerpur', 'Sattur', 'Sugauli', 'Lumding', 'Vandavasi', 'Titlagarh', 'Uchgaon', 'Mokokchung', 'Paschim Punropara', 'Sagwara', 'Ramganj Mandi', 'Tarakeswar', 'Mahalingapura', 'Dharmanagar', 'Mahemdabad', 'Manendragarh', 'Uran', 'Tharamangalam', 'Tirukkoyilur', 'Pen', 'Makhdumpur', 'Maner', 'Oddanchatram', 'Palladam', 'Mundi', 'Nabarangapur', 'Mudalagi', 'Samalkha', 'Nepanagar', 'Karjat', 'Ranavav', 'Pedana', 'Pinjore', 'Lakheri', 'Pasan', 'Puttur', 'Vadakkuvalliyur', 'Tirukalukundram', 'Mahidpur', 'Mussoorie', 'Muvattupuzha', 'Rasra', 'Udaipurwati', 'Manwath', 'Adoor', 'Uthamapalayam', 'Partur', 'Nahan', 'Ladwa', 'Mankachar', 'Nongstoin', 'Losal', 'Sri Madhopur', 'Ramngarh', 'Mavelikkara', 'Rawatsar', 'Rajakhera', 'Lar', 'Lal Gopalganj Nindaura', 'Muddebihal', 'Sirsaganj', 'Shahpura', 'Surandai', 'Sangole', 'Pavagada', 'Tharad', 'Mansa', 'Umbergaon', 'Mavoor', 'Nalbari', 'Talaja', 'Malur', 'Mangrulpir', 'Soro', 'Shahpura', 'Vadnagar', 'Raisinghnagar', 'Sindhagi', 'Sanduru', 'Sohna', 'Manavadar', 'Pihani', 'Safidon', 'Risod', 'Rosera', 'Sankari', 'Malpura', 'Sonamukhi', 'Shamsabad, Agra', 'Nokha', 'PandUrban Agglomeration', 'Mainaguri', 'Afzalpur', 'Shirur', 'Salaya', 'Shenkottai', 'Pratapgarh', 'Vadipatti', 'Nagarkurnool', 'Savner', 'Sasvad', 'Rudrapur', 'Soron', 'Sholingur', 'Pandharkaoda', 'Perumbavoor', 'Maddur', 'Nadbai', 'Talode', 'Shrigonda', 'Madhugiri', 'Tekkalakote', 'Seoni-Malwa', 'Shirdi', 'SUrban Agglomerationr', 'Terdal', 'Raver', 'Tirupathur', 'Taraori', 'Mukhed', 'Manachanallur', 'Rehli', 'Sanchore', 'Rajura', 'Piro', 'Mudabidri', 'Vadgaon Kasba', 'Vijapur', 'Viswanatham', 'Polur', 'Panagudi', 'Manawar', 'Tehri', 'Samdhan', 'Pardi', 'Rahatgarh', 'Panagar', 'Uthiramerur', 'Tirora', 'Rangia', 'Sahjanwa', 'Wara Seoni', 'Magadi', 'Rajgarh (Alwar)', 'Rafiganj', 'Tarana', 'Rampur Maniharan', 'Sheoganj', 'Raikot', 'Pauri', 'Sumerpur', 'Navalgund', 'Shahganj', 'Marhaura', 'Tulsipur', 'Sadri', 'Thiruthuraipoondi', 'Shiggaon', 'Pallapatti', 'Mahendragarh', 'Sausar', 'Ponneri', 'Mahad', 'Lohardaga', 'Tirwaganj', 'Margherita', 'Sundarnagar', 'Rajgarh', 'Mangaldoi', 'Renigunta', 'Longowal', 'Ratia', 'Lalgudi', 'Shrirangapattana', 'Niwari', 'Natham', 'Unnamalaikadai', 'PurqUrban Agglomerationzi', 'Shamsabad, Farrukhabad', 'Mirganj', 'Todaraisingh', 'Warhapur', 'Rajam', 'Urmar Tanda', 'Lonar', 'Powayan', 'P.N.Patti', 'Palampur', 'Srisailam Project (Right Flank Colony) Township', 'Sindagi', 'Sandi', 'Vaikom', 'Malda', 'Tharangambadi', 'Sakaleshapura', 'Lalganj', 'Malkangiri', 'Rapar', 'Mauganj', 'Todabhim', 'Srinivaspur', 'Murliganj', 'Reengus', 'Sawantwadi', 'Tittakudi', 'Lilong', 'Rajaldesar', 'Pathardi', 'Achhnera', 'Pacode', 'Naraura', 'Nakur', 'Palai', 'Morinda, India', 'Manasa', 'Nainpur', 'Sahaspur', 'Pauni', 'Prithvipur', 'Ramtek', 'Silapathar', 'Songadh', 'Safipur', 'Sohagpur', 'Mul', 'Sadulshahar', 'Phillaur', 'Sambhar', 'Prantij', 'Nagla', 'Pattran', 'Mount Abu', 'Reoti', 'Tenu dam-cum-Kathhara', 'Panchla', 'Sitarganj', 'Pasighat', 'Motipur', "O' Valley", 'Raghunathpur', 'Suriyampalayam', 'Qadian', 'Rairangpur', 'Silvassa', 'Nowrozabad (Khodargama)', 'Mangrol', 'Soyagaon', 'Sujanpur', 'Manihari', 'Sikanderpur', 'Mangalvedhe', 'Phulera', 'Ron', 'Sholavandan', 'Saidpur', 'Shamgarh', 'Thammampatti', 'Maharajpur', 'Multai', 'Mukerian', 'Sirsi', 'Purwa', 'Sheohar', 'Namagiripettai', 'Parasi', 'Lathi', 'Lalganj', 'Narkhed', 'Mathabhanga', 'Shendurjana', 'Peravurani', 'Mariani', 'Phulpur', 'Rania', 'Pali', 'Pachore', 'Parangipettai', 'Pudupattinam', 'Panniyannur', 'Maharajganj', 'Rau', 'Monoharpur', 'Mandawa', 'Marigaon', 'Pallikonda', 'Pindwara', 'Shishgarh', 'Patur', 'Mayang Imphal', 'Mhowgaon', 'Guruvayoor', 'Mhaswad', 'Sahawar', 'Sivagiri', 'Mundargi', 'Punjaipugalur', 'Kailasahar', 'Samthar', 'Sakti', 'Sadalagi', 'Silao', 'Mandalgarh', 'Loha', 'Pukhrayan', 'Padmanabhapuram', 'Belonia', 'Saiha', 'Srirampore', 'Talwara', 'Puthuppally', 'Khowai', 'Vijaypur', 'Takhatgarh', 'Thirupuvanam', 'Adra', 'Piriyapatna', 'Obra', 'Adalaj', 'Nandgaon', 'Barh', 'Chhapra', 'Panamattom', 'Niwai', 'Bageshwar', 'Tarbha', 'Adyar', 'Narsinghgarh', 'Warud', 'Asarganj','Sarsod']





splitlines = text.splitlines( )

def getInvoicenumber(text):
    invoicenumber = [] 
   

    inv = re.compile(r'(receipt|receipt number|receipt no|invoice number|invoice no|inv|invoice|bill no|bill|bill id|invno|billid|billno|invoicenumber|invoiceno)\s*([:.-]+)?\s*([a-zA-Z0-9/\.-]+[\d])',re.IGNORECASE)

    for j in range(len(splitlines)):
        splitlines[j]=splitlines[j].replace("\\", "") 
    for i in inv.finditer(splitlines[j]):
        invoicenumber.append(i.group(3))
    return invoicenumber    
        
invoicenumber = getInvoicenumber(text) 
print(invoicenumber)



def getname(splitlines):
    name=''
    for j in range(len(splitlines)):
        # print (splitlines)

        x = re.search(r"(pvt ltd|pvt|ltd|pvt. ltd|limited|llp)",splitlines[j],re.IGNORECASE)
        url = re.search(r'(www|WWW).[a-zA-Z0-9\.]*\b', splitlines[j])


        if (x):
            name = x.string
            nameline=j
            break

        elif(url):
                #print (splitlines[j])	
                website=url.group(0)	
               # print(url.group(0))
                webarray = re.split("\.", website)
                index=website.index('www')
                name=webarray[index+1]
                break
        else:
            name=""
            nameline=2


    i=0
    flag=0
    while(flag!=1 and name==''):
        firstline=re.search(r"(tax|invoice|tax invoice|,|Return|exchange|thankyou|nvoice)",splitlines[i],re.IGNORECASE)
        if(firstline or splitlines[i]=="" or splitlines[i]==" " or splitlines[i]=="\t"):
            i=i+1
            # print("enter if")
        else:
            flag=1
            name=splitlines[i]
            nameline=i
    return [name,nameline]
            # print("enter else")

invoicename = getname(splitlines)
print(invoicename)
nameline=invoicename[1]

def getaddress(splitlines,nameline):
    i=0
    city=0
    address = []
    for inc in range(nameline+1,nameline+15):
        # print(nameline+i+1)
        # print(splitlines[inc])
        for city in test:
            # print(city,splitlines[nameline+i+1])
            cityfound=re.search(city,splitlines[nameline+i+1],re.IGNORECASE)
            if(cityfound):
                # print("city found at line :",nameline+i+1)
                break
        if(cityfound):
            # print("city found at line :",nameline+i+1)
            break
        else:
            i=i+1

    for add in range(nameline+1,nameline+i+2):
        address.append(splitlines[add])
    return address
invoiceaddress = getaddress(splitlines,nameline)
print(invoiceaddress)


ta=re.compile(r'\d*\.[0-9][0-9]')
f= open("text.txt","r")
with open('text.txt', 'r') as myfile:
    data= f.read()
matches = ta.findall(data)

results = list(map(float, matches))

file=open("text.txt",'r')
f=file.readlines()
file.close()


def getAmount(f):
    i=0
    index=[]
    val=[]
    for line in f:
        ta=re.compile(r'\d*\.[0-9][0-9]').findall(line)
        if ta!=[]:
            index.append(i)
            val.append(max(ta))
        i=i+1

    for item in val:
        item=float(item)

    max_amt=max(map(float, val))
    max_amt='%.2f'%max_amt 
    ind=val.index(max_amt)
    ind2=index[ind]
    max_line=f[ind2]
    patterns = ['Net Amt','Total','Net Amount']
    flag=0
    for pattern in patterns:
        if re.search(pattern, max_line,re.IGNORECASE):
            flag=1
        else:
            flag=0

    if flag==1:
        total_amount=max_amt
    else:
        haha = list(map(float, val))
        r=max(haha)
        haha.remove(r)
        total_amount=max(haha)
    
    return total_amount


invoiceamount = getAmount(f)
print(invoiceamount)  
df = pd.DataFrame(invoicename, columns=['Store_Name'])
df['Store_Address'] = invoiceaddress
df['Total_Amount'] = invoiceamount
df['Invoice_Number'] = invoicenumber
df['Invoice_Date'] = invoice_date
df.to_csv('Dataextract.csv') 

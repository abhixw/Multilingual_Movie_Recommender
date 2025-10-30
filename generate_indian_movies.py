import random
from datetime import datetime, timedelta

# Regional language components with more authentic titles
regional_titles = {
    'Hindi': {
        'prefixes': ['Dil ', 'Pyaar ', 'Ek ', 'Main ', 'Mere ', 'Tum ', 'Zindagi ', 'Ishq ', 'Kuch ', 'Aaj ', 
                    'Phir ', 'Mohabbat ', 'Kabhi ', 'Jab ', 'Hum ', 'Teri ', 'Meri ', 'Do ', 'Teen ', 'Char '],
        'suffixes': ['Safar', 'Kahani', 'Raaz', 'Dastaan', 'Zindagi', 'Mohabbat', 'Ishq', 'Roshni', 'Yaadein', 
                    'Bandhan', 'Khiladi', 'Baazi', 'Jung', 'Jeet', 'Haar', 'Pyar', 'Dil', 'Ghar', 'Bhool', 'Yaad']
    },
    'Tamil': {
        'prefixes': ['Thalaiva ', 'Kaadhal ', 'Kannu ', 'Urimai ', 'Thamizh ', 'Nenjam ', 'Thiru ', 'Aayirathil ', 
                    'Mudhal ', 'Kadal ', 'Vaanam ', 'Bhoomi ', 'Manithan ', 'Megam ', 'Poo ', 'Ponnu ', 'Raaja ', 
                    'Vettai ', 'Thirupathi ', 'Pudhiya '],
        'suffixes': ['Kavithai', 'Kadhalan', 'Manam', 'Vaazhkai', 'Vaali', 'Thalaivan', 'Nanban', 'Kadhal', 
                    'Neram', 'Bhoomi', 'Veeran', 'Singam', 'Roja', 'Mazhai', 'Minnal', 'Muthu', 'Raja', 'Velai', 
                    'Ponnu', 'Nila']
    },
    'Telugu': {
        'prefixes': ['Prema ', 'Naa ', 'Mana ', 'Oka ', 'Iddaru ', 'Nuvvu ', 'Nenu ', 'Meeru ', 'Raja ', 'Adavi ',
                    'Pedda ', 'Chinna ', 'Maha ', 'Swarna ', 'Koti ', 'Bangaru ', 'Simha ', 'Veera ', 'Yuddham ', 
                    'Bhale '],
        'suffixes': ['Katha', 'Lokam', 'Premalo', 'Manasu', 'Gunde', 'Rahasyam', 'Bandham', 'Prema', 'Rekkalu', 
                    'Cheruvu', 'Simham', 'Raju', 'Veerudu', 'Koduku', 'Pellam', 'Abbayi', 'Ammayi', 'Rowdy', 
                    'Police', 'Rangam']
    },
    'Malayalam': {
        'prefixes': ['Oru ', 'Njan ', 'Ente ', 'Njanum ', 'Aaru ', 'Ee ', 'Namma ', 'Priya ', 'Mera ', 'Kali ',
                    'Kadal ', 'Manja ', 'Puthiya ', 'Pazhaya ', 'Innale ', 'Nale ', 'Swarna ', 'Megha ', 'Varna ', 
                    'Aana '],
        'suffixes': ['Kadha', 'Premam', 'Jeevitham', 'Sneham', 'Lokam', 'Samayam', 'Dinam', 'Veedu', 'Mazha', 
                    'Neram', 'Kochu', 'Periya', 'Makan', 'Mol', 'Ammavan', 'Kalyanam', 'Yathra', 'Nadodi', 
                    'Thaaram', 'Chemmeen']
    },
    'Bengali': {
        'prefixes': ['Ek ', 'Amar ', 'Tomar ', 'Prothom ', 'Shesh ', 'Bondhu ', 'Jibon ', 'Mon ', 'Shudhu ', 'Ami ',
                    'Tumi ', 'Shei ', 'Aaj ', 'Kaal ', 'Prem ', 'Raat ', 'Din ', 'Shotti ', 'Mithya ', 'Noya '],
        'suffixes': ['Bhalobasha', 'Golpo', 'Jibon', 'Raat', 'Prohor', 'Shokal', 'Shondhya', 'Aaloy', 'Bristi', 
                    'Kotha', 'Gaan', 'Naach', 'Chobi', 'Manush', 'Nogor', 'Gram', 'Nodi', 'Phool', 'Pakhi', 'Megh']
    },
    'Marathi': {
        'prefixes': ['Maza ', 'Tumcha ', 'Aamcha ', 'Ekach ', 'Prema ', 'Nav ', 'Man ', 'Dil ', 'Sukh ', 'Jiv ',
                    'Swapna ', 'Rang ', 'Prem ', 'Baal ', 'Ek ', 'Don ', 'Teen ', 'Mee ', 'Tu ', 'Mi '],
        'suffixes': ['Gaon', 'Prem', 'Ahe', 'Maitri', 'Sathi', 'Khel', 'Vishwas', 'Dhaga', 'Rang', 'Maza',
                    'Mulgi', 'Mulga', 'Ghar', 'Kunku', 'Paus', 'Tuza', 'Aga', 'Maher', 'Sasu', 'Navra']
    },
    'Punjabi': {
        'prefixes': ['Mera ', 'Tera ', 'Pyaar ', 'Dil ', 'Zindagi ', 'Yaar ', 'Jatt ', 'Putt ', 'Singh ', 'Munde ',
                    'Kudiyan ', 'Pind ', 'Vehla ', 'Geet ', 'Jugni ', 'Desi ', 'Punjabi ', 'Vadda ', 'Chota ', 
                    'Mere '],
        'suffixes': ['Pind', 'Sardar', 'Mutiyaar', 'Pyaar', 'Dil', 'Gabru', 'Sardaar', 'Jaan', 'Mera', 'Yaari',
                    'Yaar', 'Jatt', 'Babe', 'Naale', 'Jodi', 'Bhaag', 'Jawani', 'Ishq', 'Kudiyan', 'Munde']
    },
    'Kannada': {
        'prefixes': ['Nanna ', 'Ninna ', 'Prema ', 'Jeeva ', 'Hrudaya ', 'Surya ', 'Krishna ', 'Mana ', 'Kannu ', 
                    'Gelathi ', 'Huduga ', 'Hudugi ', 'Ranga ', 'Gana ', 'Banna ', 'Hemmeya ', 'Namma ', 'Ond ', 
                    'Eradu ', 'Muru '],
        'suffixes': ['Kathe', 'Preethi', 'Sangama', 'Jeevana', 'Geethe', 'Belaku', 'Kanasu', 'Jodi', 'Ranga', 
                    'Habba', 'Huduga', 'Naanu', 'Yodha', 'Rajkumar', 'Ganda', 'Hejje', 'Hoovu', 'Male', 'Bisilu', 
                    'Belli']
    }
}

# Expanded genres with cultural context
genres = {
    'Drama': 15,
    'Comedy': 12,
    'Romance': 12,
    'Action': 10,
    'Family': 8,
    'Social': 8,
    'Musical': 6,
    'Thriller': 5,
    'Historical': 5,
    'Devotional': 4,
    'War': 3,
    'Sports': 3,
    'Political': 3,
    'Mythology': 3,
    'Crime': 3,
    'Horror': 2,
    'Biographical': 2,
    'Folk': 2,
    'Art House': 1,
    'Documentary': 1
}

# Cultural festivals and events
festivals = ['Diwali', 'Holi', 'Pongal', 'Durga Puja', 'Ganesh Chaturthi', 'Onam', 'Baisakhi', 
            'Navratri', 'Eid', 'Christmas', 'Sankranti', 'Ugadi', 'Bihu', 'Vishu']

def generate_year():
    """Generate year with more weight to recent years"""
    weights = [1] * 30 + [2] * 20 + [3] * 15 + [4] * 10 + [5] * 5
    year_range = list(range(1950, 2026))
    return random.choices(year_range, weights=weights[:len(year_range)])[0]

def generate_movie_title(language):
    """Generate culturally authentic movie title"""
    titles = regional_titles[language]
    prefix = random.choice(titles['prefixes'])
    suffix = random.choice(titles['suffixes'])
    
    # Sometimes add festival/cultural references
    if random.random() < 0.05:  # 5% chance
        festival = random.choice(festivals)
        return f"{prefix}{festival}"
    
    # Sometimes add numbers
    if random.random() < 0.03:  # 3% chance
        number = random.choice(['100', '786', '420', '99', '3', '7', '21'])
        return f"{prefix}{suffix} {number}"
    
    return f"{prefix}{suffix}"

def generate_genres():
    """Select 1-3 genres based on weighted probabilities"""
    num_genres = random.choices([1, 2, 3], weights=[20, 60, 20])[0]
    selected_genres = random.choices(list(genres.keys()), 
                                  weights=list(genres.values()),
                                  k=num_genres)
    return '|'.join(sorted(set(selected_genres)))

# Calculate movies per language based on industry size (scaled to 60,000 total)
language_distribution = {
    'Hindi': 20000,    # Bollywood - Largest film industry
    'Tamil': 9000,     # Kollywood - Major South Indian industry
    'Telugu': 9000,    # Tollywood - Major South Indian industry
    'Malayalam': 6000, # Mollywood - Kerala film industry
    'Kannada': 6000,   # Sandalwood - Karnataka film industry
    'Bengali': 4000,   # Tollywood (Bengal) - West Bengal film industry
    'Marathi': 3000,   # M-Town - Maharashtra regional cinema
    'Punjabi': 3000,   # Pollywood - Punjab film industry
}

# Generate movies
movies = []
movie_id = 1

print("Generating movies for each language...")
for language, count in language_distribution.items():
    print(f"  Generating {count} {language} movies...")
    for _ in range(count):
        year = generate_year()
        title = generate_movie_title(language)
        movie_genres = generate_genres()
        # Format: Title (Year) [Language]
        full_title = f"{title} ({year}) [{language}]"
        movies.append(f"{movie_id}::{full_title}::{movie_genres}")
        movie_id += 1

# Shuffle movies to mix languages
random.shuffle(movies)

# Write to movies.dat
with open('movies.dat', 'w', encoding='utf-8') as f:
    f.write('\n'.join(movies))

print(f"\n✅ Successfully generated {len(movies)} Indian movies across {len(language_distribution)} languages!")
print("\nLanguage Distribution:")
for lang, count in sorted(language_distribution.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(movies)) * 100
    print(f"  {lang:12} : {count:6,} movies ({percentage:5.1f}%)")
print(f"\n📁 Saved to: movies.dat")
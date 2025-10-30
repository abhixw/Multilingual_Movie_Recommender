import random
import time
from datetime import datetime, timedelta

# Parameters for generating ratings
NUM_USERS = 5000        # Number of users
NUM_MOVIES = 60000      # Total number of Indian movies (updated to match generation script)
MIN_RATINGS_PER_USER = 20    # Minimum number of movies each user rates
MAX_RATINGS_PER_USER = 200   # Maximum number of movies each user rates

# Language preferences - simulate real-world user preferences
# Users tend to prefer certain languages based on region
LANGUAGE_REGIONS = {
    'Hindi': 0.40,      # 40% of users prefer Hindi (national reach)
    'Tamil': 0.12,      # 12% prefer Tamil
    'Telugu': 0.12,     # 12% prefer Telugu
    'Malayalam': 0.08,  # 8% prefer Malayalam
    'Kannada': 0.08,    # 8% prefer Kannada
    'Bengali': 0.07,    # 7% prefer Bengali
    'Marathi': 0.07,    # 7% prefer Marathi
    'Punjabi': 0.06,    # 6% prefer Punjabi
}

# Movie ID ranges per language (based on 60,000 movies)
# These ranges help us target language-specific movies
MOVIE_RANGES = {
    'Hindi': (1, 20000),
    'Tamil': (20001, 29000),
    'Telugu': (29001, 38000),
    'Malayalam': (38001, 44000),
    'Kannada': (44001, 50000),
    'Bengali': (50001, 54000),
    'Marathi': (54001, 57000),
    'Punjabi': (57001, 60000),
}

def assign_user_language_preference():
    """Assign a primary language preference to a user"""
    languages = list(LANGUAGE_REGIONS.keys())
    weights = list(LANGUAGE_REGIONS.values())
    return random.choices(languages, weights=weights)[0]

def get_movies_for_user(preferred_language, num_ratings):
    """
    Generate movie IDs for a user based on their language preference.
    80% from preferred language, 20% from other languages (exploration)
    """
    movies = []
    
    # Get preferred language movies (80%)
    preferred_count = int(num_ratings * 0.80)
    lang_range = MOVIE_RANGES[preferred_language]
    preferred_movies = random.sample(
        range(lang_range[0], lang_range[1] + 1), 
        min(preferred_count, lang_range[1] - lang_range[0] + 1)
    )
    movies.extend(preferred_movies)
    
    # Get movies from other languages (20% - exploration)
    remaining_count = num_ratings - len(movies)
    if remaining_count > 0:
        # Exclude preferred language range
        other_movie_ids = []
        for lang, (start, end) in MOVIE_RANGES.items():
            if lang != preferred_language:
                other_movie_ids.extend(range(start, end + 1))
        
        if len(other_movie_ids) >= remaining_count:
            other_movies = random.sample(other_movie_ids, remaining_count)
            movies.extend(other_movies)
        else:
            movies.extend(other_movie_ids)
    
    return movies

def generate_rating_with_context(movie_id, user_preferred_language):
    """
    Generate rating considering whether movie matches user's language preference.
    Movies in preferred language get slightly higher ratings on average.
    """
    # Check if movie is in user's preferred language
    is_preferred_lang = False
    lang_range = MOVIE_RANGES[user_preferred_language]
    if lang_range[0] <= movie_id <= lang_range[1]:
        is_preferred_lang = True
    
    if is_preferred_lang:
        # Slightly higher ratings for preferred language
        rating = random.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 5, 4])[0]
    else:
        # Normal distribution for other languages
        rating = random.choices([1, 2, 3, 4, 5], weights=[1, 2, 3, 4, 3])[0]
    
    return rating

# Generate ratings
print("Generating ratings...")
ratings = []
current_timestamp = int(time.time())
one_day = 24 * 60 * 60  # seconds in a day

# Track statistics
language_stats = {lang: {'ratings': 0, 'total_rating': 0} for lang in LANGUAGE_REGIONS.keys()}

# For each user
for user_id in range(1, NUM_USERS + 1):
    if user_id % 500 == 0:
        print(f"  Processing user {user_id}/{NUM_USERS}...")
    
    # Assign language preference to user
    user_preferred_language = assign_user_language_preference()
    
    # Decide how many movies this user will rate
    num_ratings = random.randint(MIN_RATINGS_PER_USER, MAX_RATINGS_PER_USER)
    
    # Get movies based on user's language preference
    movies_to_rate = get_movies_for_user(user_preferred_language, num_ratings)
    
    # Add some bias towards rating newer movies (simulate recent releases getting more attention)
    if random.random() < 0.6:  # 60% of users will have this bias
        # Add recent movies from their preferred language
        lang_range = MOVIE_RANGES[user_preferred_language]
        recent_count = random.randint(5, 15)
        recent_start = max(lang_range[0], lang_range[1] - 500)  # Last 500 movies in language
        additional_recent = random.sample(
            range(recent_start, lang_range[1] + 1),
            min(recent_count, lang_range[1] - recent_start + 1)
        )
        movies_to_rate.extend(additional_recent)
        movies_to_rate = list(set(movies_to_rate))  # Remove duplicates
    
    for movie_id in movies_to_rate:
        # Generate rating based on context
        rating = generate_rating_with_context(movie_id, user_preferred_language)
        
        # Generate timestamp (within last 3 years, more recent = more likely)
        # Exponential distribution favoring recent ratings
        days_ago = int(random.expovariate(1/365))  # Average 1 year ago
        days_ago = min(days_ago, 365 * 3)  # Cap at 3 years
        timestamp = current_timestamp - (days_ago * one_day)
        
        # Track statistics
        for lang, (start, end) in MOVIE_RANGES.items():
            if start <= movie_id <= end:
                language_stats[lang]['ratings'] += 1
                language_stats[lang]['total_rating'] += rating
                break
        
        # Add rating
        ratings.append(f"{user_id}::{movie_id}::{rating}::{timestamp}")

# Sort ratings by user_id and timestamp (more realistic)
ratings.sort(key=lambda x: (int(x.split("::")[0]), int(x.split("::")[3])))

# Write to ratings.dat
print("\nWriting to ratings.dat...")
with open('ratings.dat', 'w') as f:
    f.write('\n'.join(ratings))

# Print statistics
print(f"\n✅ Successfully generated {len(ratings):,} ratings from {NUM_USERS:,} users!")
print(f"\nRating Statistics by Language:")
print(f"{'Language':<12} | {'Ratings':>10} | {'Avg Rating':>10} | {'Percentage':>10}")
print("-" * 50)

for lang in sorted(language_stats.keys(), key=lambda x: language_stats[x]['ratings'], reverse=True):
    stats = language_stats[lang]
    count = stats['ratings']
    avg_rating = stats['total_rating'] / count if count > 0 else 0
    percentage = (count / len(ratings)) * 100
    print(f"{lang:<12} | {count:>10,} | {avg_rating:>10.2f} | {percentage:>9.1f}%")

print(f"\n📁 Saved to: ratings.dat")
print(f"\nAverage ratings per user: {len(ratings) / NUM_USERS:.1f}")
print(f"Date range: {datetime.fromtimestamp(min(int(r.split('::')[3]) for r in ratings)).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(current_timestamp).strftime('%Y-%m-%d')}")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict
import uuid

# Configuration
START_DATE = "2026-01-01"
END_DATE = "2026-01-30"
KERALA_DISTRICTS = ["Thiruvananthapuram", "Ernakulam", "Kozhikode", "Thrissur", "Kannur"]
JOB_CATEGORIES = ["Plumber", "Electrician", "Carpenter", "Painter", "AC Technician"]
COMMISSION_RATE = 0.001  # 0.1%

class BeinerUser:
    """Class to represent a user with their metrics"""
    
    def __init__(self, user_id: str, user_type: str, registration_date: str):
        self.user_id = user_id
        self.user_type = user_type
        self.registration_date = registration_date
        
        # Determine account status with probabilities
        status_prob = random.random()
        if status_prob < 0.7:  # 70% Active
            self.account_status = "Active"
        elif status_prob < 0.85:  # 15% Inactive
            self.account_status = "Inactive"
        else:  # 15% Suspended
            self.account_status = "Suspended"
        
        # Initialize metrics
        self.total_bookings = 0
        self.completed_bookings = 0
        self.cancelled_bookings = 0
        self.total_spent = 0.0
        self.total_earned = 0.0
        self.platform_commission = 0.0
        self.avg_rating = 0.0
        self.total_reviews = 0
        self.last_active = None
        
        # Calculate initial metrics based on registration date
        self._initialize_metrics(registration_date)
    
    def _initialize_metrics(self, reg_date: str):
        """Initialize user metrics based on registration date"""
        reg_datetime = datetime.strptime(reg_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(END_DATE, '%Y-%m-%d')
        days_active = max(1, (end_datetime - reg_datetime).days)
        
        if self.user_type == 'worker':
            # Workers have bookings and earnings
            if self.account_status == 'Active':
                # Active workers have bookings
                self.total_bookings = random.randint(0, min(20, days_active * 2))
                self.completed_bookings = int(self.total_bookings * random.uniform(0.4, 0.9))
                self.cancelled_bookings = self.total_bookings - self.completed_bookings
                
                if self.completed_bookings > 0:
                    self.total_earned = round(self.completed_bookings * random.randint(500, 1500), 2)
                    self.platform_commission = round(self.total_earned * COMMISSION_RATE, 3)
                    
                    # Generate rating and reviews for workers with bookings
                    if self.completed_bookings > 3:
                        self.avg_rating = round(random.uniform(3.0, 5.0), 1)
                        self.total_reviews = random.randint(1, self.completed_bookings // 2)
                    else:
                        self.avg_rating = 0.0
                        self.total_reviews = 0
        else:  # employer
            if self.account_status == 'Active':
                # Active employers have bookings and spending
                self.total_bookings = random.randint(0, min(15, days_active))
                self.completed_bookings = int(self.total_bookings * random.uniform(0.4, 0.85))
                self.cancelled_bookings = self.total_bookings - self.completed_bookings
                
                if self.completed_bookings > 0:
                    self.total_spent = round(self.completed_bookings * random.randint(1000, 2000), 2)
                    self.platform_commission = round(self.total_spent * COMMISSION_RATE, 3)
                    self.total_reviews = random.randint(0, self.completed_bookings)
    
    def update_last_active(self):
        """Update the last active timestamp"""
        end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
        reg_date = datetime.strptime(self.registration_date, '%Y-%m-%d')
        
        # Last active should be between registration date and end date
        if self.account_status == "Active":
            # Active users are more recently active
            days_range = (end_date - reg_date).days
            if days_range > 0:
                days_offset = random.randint(0, days_range // 2)  # Recently active
                self.last_active = (end_date - timedelta(days=days_offset)).strftime('%Y-%m-%d %H:%M:%S')
            else:
                self.last_active = reg_date.strftime('%Y-%m-%d %H:%M:%S')
        elif self.account_status == "Inactive":
            # Inactive users haven't been active for a while
            days_range = (end_date - reg_date).days
            if days_range > 7:
                days_offset = random.randint(days_range // 2, days_range - 1)
                self.last_active = (end_date - timedelta(days=days_offset)).strftime('%Y-%m-%d %H:%M:%S')
            else:
                self.last_active = reg_date.strftime('%Y-%m-%d %H:%M:%S')
        else:  # Suspended
            # Suspended users were active recently before suspension
            days_range = (end_date - reg_date).days
            if days_range > 3:
                days_offset = random.randint(1, min(3, days_range))
                self.last_active = (end_date - timedelta(days=days_offset)).strftime('%Y-%m-%d %H:%M:%S')
            else:
                self.last_active = reg_date.strftime('%Y-%m-%d %H:%M:%S')
    
    def to_dict(self) -> Dict:
        """Convert user to dictionary format matching sample data"""
        timestamp = datetime.strptime(END_DATE, '%Y-%m-%d').strftime('%Y-%m-%d 21:54:57')
        
        return {
            'timestamp': timestamp,
            'user_id': self.user_id,
            'user_type': 'worker' if self.user_type == 'worker' else 'employer',
            'registration_date': self.registration_date,
            'account_status': self.account_status,
            'total_bookings': self.total_bookings,
            'completed_bookings': self.completed_bookings,
            'cancelled_bookings': self.cancelled_bookings,
            'total_spent': round(self.total_spent, 2),
            'total_earned': round(self.total_earned, 2),
            'platform_commission': round(self.platform_commission, 3),
            'avg_rating': self.avg_rating,
            'total_reviews': self.total_reviews,
            'last_active': self.last_active
        }

def generate_user_id(user_type: str, index: int) -> str:
    """Generate user ID in format WK0001 or EM0001"""
    prefix = "WK" if user_type == "worker" else "EM"
    return f"{prefix}{str(index).zfill(4)}"

def create_users() -> List[BeinerUser]:
    """Create users with realistic distribution"""
    users = []
    
    # Create workers (more workers than employers)
    num_workers = random.randint(20, 30)
    for i in range(1, num_workers + 1):
        user_id = generate_user_id("worker", i)
        
        # Registration dates distributed throughout the month
        start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
        days_offset = random.randint(0, 29)  # Within January
        registration_date = (start_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        
        user = BeinerUser(user_id, "worker", registration_date)
        user.update_last_active()
        users.append(user)
    
    # Create employers
    num_employers = random.randint(8, 15)
    for i in range(1, num_employers + 1):
        user_id = generate_user_id("employer", i)
        
        start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
        days_offset = random.randint(0, 29)
        registration_date = (start_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        
        user = BeinerUser(user_id, "employer", registration_date)
        user.update_last_active()
        users.append(user)
    
    return users

def validate_dataset(df: pd.DataFrame):
    """Validate the dataset matches requirements"""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    
    # Check required columns
    required_columns = ['timestamp', 'user_id', 'user_type', 'registration_date', 
                       'account_status', 'total_bookings', 'completed_bookings', 
                       'cancelled_bookings', 'total_spent', 'total_earned', 
                       'platform_commission', 'avg_rating', 'total_reviews', 'last_active']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"ERROR: Missing columns: {missing_columns}")
    else:
        print("✓ All required columns present")
    
    # Check data types
    print(f"\nData types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Check unique users
    print(f"\nUnique users: {df['user_id'].nunique()}")
    
    # Check user type distribution
    print(f"\nUser type distribution:")
    user_type_counts = df['user_type'].value_counts()
    for user_type, count in user_type_counts.items():
        print(f"  {user_type}: {count} ({count/len(df)*100:.1f}%)")
    
    # Check account status distribution
    print(f"\nAccount status distribution:")
    status_counts = df['account_status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count} ({count/len(df)*100:.1f}%)")
    
    # Validate commission calculation
    print(f"\nCommission validation:")
    for idx, row in df.iterrows():
        if row['user_type'] == 'worker' and row['total_earned'] > 0:
            expected_commission = round(row['total_earned'] * COMMISSION_RATE, 3)
            if abs(row['platform_commission'] - expected_commission) > 0.001:
                print(f"  Warning: Worker {row['user_id']} commission mismatch")
        elif row['user_type'] == 'employer' and row['total_spent'] > 0:
            expected_commission = round(row['total_spent'] * COMMISSION_RATE, 3)
            if abs(row['platform_commission'] - expected_commission) > 0.001:
                print(f"  Warning: Employer {row['user_id']} commission mismatch")
    
    print("✓ Commission calculation validated")
    
    # Validate booking consistency
    print(f"\nBooking validation:")
    inconsistent = df[df['total_bookings'] != (df['completed_bookings'] + df['cancelled_bookings'])]
    if len(inconsistent) > 0:
        print(f"  Warning: {len(inconsistent)} users have inconsistent booking counts")
    else:
        print("✓ All booking counts are consistent")

def generate_dataset():
    """Generate the complete dataset"""
    print("BEINER PLATFORM DATASET GENERATOR")
    print("=" * 60)
    print(f"Time period: {START_DATE} to {END_DATE}")
    print(f"Commission rate: {COMMISSION_RATE*100}%")
    print("=" * 60)
    
    # Create users
    print("Creating users...")
    users = create_users()
    
    # Convert to DataFrame
    print("Creating dataset...")
    records = [user.to_dict() for user in users]
    df = pd.DataFrame(records)
    
    # Sort by user_id
    df = df.sort_values('user_id').reset_index(drop=True)
    
    # Ensure timestamp is consistent (all records have same timestamp as in sample)
    df['timestamp'] = datetime.strptime(END_DATE, '%Y-%m-%d').strftime('%Y-%m-%d 21:54:57')
    
    # Validate the dataset
    validate_dataset(df)
    
    return df

def save_dataset(df: pd.DataFrame, filename: str = "beiner_dataset.csv"):
    """Save dataset to CSV"""
    print(f"\nSaving dataset to {filename}...")
    df.to_csv(filename, index=False)
    
    # Calculate file size
    import os
    file_size = os.path.getsize(filename) / 1024
    print(f"File size: {file_size:.2f} KB")
    
    return filename

def print_sample_data(df: pd.DataFrame, num_samples: int = 10):
    """Print sample data matching your format"""
    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    
    # Display sample in the exact format you provided
    sample_df = df.head(num_samples).copy()
    
    # Format the display exactly like your sample
    print("timestamp,user_id,user_type,registration_date,account_status,total_bookings,completed_bookings,cancelled_bookings,total_spent,total_earned,platform_commission,avg_rating,total_reviews,last_active")
    
    for _, row in sample_df.iterrows():
        # Format each row to match your sample
        line = f"{row['timestamp']},{row['user_id']},{row['user_type']},{row['registration_date']},{row['account_status']},{row['total_bookings']},{row['completed_bookings']},{row['cancelled_bookings']},{row['total_spent']},{row['total_earned']},{row['platform_commission']},{row['avg_rating']},{row['total_reviews']},{row['last_active']}"
        print(line)

def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"Total users: {len(df)}")
    print(f"Total workers: {len(df[df['user_type'] == 'worker'])}")
    print(f"Total employers: {len(df[df['user_type'] == 'employer'])}")
    
    print(f"\nOverall statistics:")
    print(f"Total bookings: {df['total_bookings'].sum()}")
    print(f"Total completed bookings: {df['completed_bookings'].sum()}")
    print(f"Total cancelled bookings: {df['cancelled_bookings'].sum()}")
    print(f"Total spent by employers: ₹{df['total_spent'].sum():,.2f}")
    print(f"Total earned by workers: ₹{df['total_earned'].sum():,.2f}")
    print(f"Total platform commission: ₹{df['platform_commission'].sum():,.2f}")
    
    print(f"\nWorker statistics:")
    workers_df = df[df['user_type'] == 'worker']
    print(f"Average bookings per worker: {workers_df['total_bookings'].mean():.1f}")
    print(f"Average earnings per worker: ₹{workers_df['total_earned'].mean():,.2f}")
    print(f"Average rating: {workers_df[workers_df['avg_rating'] > 0]['avg_rating'].mean():.1f}")
    
    print(f"\nEmployer statistics:")
    employers_df = df[df['user_type'] == 'employer']
    print(f"Average bookings per employer: {employers_df['total_bookings'].mean():.1f}")
    print(f"Average spending per employer: ₹{employers_df['total_spent'].mean():,.2f}")

def main():
    """Main execution function"""
    try:
        # Generate dataset
        df = generate_dataset()
        
        # Print sample data
        print_sample_data(df)
        
        # Print summary
        print_summary_statistics(df)
        
        # Save to CSV
        filename = save_dataset(df, "beiner_platform_data.csv")
        
        print(f"\n✓ Dataset successfully generated and saved to {filename}")
        print(f"✓ Total records: {len(df)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
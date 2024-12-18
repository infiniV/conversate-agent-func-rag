# conversate-agent-func-rag

## Overview

`conversate-agent-func-rag` is a voice-based booking assistant designed to manage venue bookings. It leverages various plugins and APIs to provide a seamless booking experience through voice interactions.

## Features

- **Voice Interaction**: Uses voice commands to interact with the booking system.
- **Booking Management**: Allows users to start a booking, check available times, view free seats, and book a seat.
- **Phone Validation**: Validates phone numbers to ensure correct format.
- **Pronunciation Formatting**: Formats text for better pronunciation during text-to-speech.
- **Cached Data**: Uses caching to improve performance when fetching available times.
- **Supabase Integration**: Stores and retrieves booking data from Supabase.

## Setup

1. **Clone the repository**:

   ```sh
   git clone https://github.com/your-repo/conversate-agent-func-rag.git
   cd conversate-agent-func-rag
   ```

2. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory and add your Supabase credentials:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

## Usage

Run the main script to start the voice-based booking assistant:

```sh
python main-agent.py
```

## Functions

- **start_booking(name: str, phone: str)**: Initializes the booking process with the user's name and phone number.
- **get_next_available_slot()**: Retrieves the next available time slot for booking.
- **get_available_times(start_time: str = None, end_time: str = None)**: Lists available time slots within a specified range.
- **get_free_seats(time: str)**: Lists available seats for a given time slot.
- **book_seat(phone: str, time: str, seat_number: int)**: Books a seat for the specified time slot.
- **view_bookings(phone: str)**: Retrieves all bookings associated with the provided phone number.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

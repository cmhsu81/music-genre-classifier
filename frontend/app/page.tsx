"use client";

import { useState, ChangeEvent } from 'react';
import axios from 'axios';

const genreDescriptions: { [key: string]: string } = {
  blues: "Blues is a music genre rooted in African-American history, known for its soulful melodies and themes of hardship and emotion.",
  classical: "Classical music spans centuries of musical tradition and is characterized by its complex orchestration and formality.",
  country: "Country music originated from American folk traditions and often features guitars, storytelling, and themes of rural life.",
  disco: "Disco emerged in the 1970s with upbeat rhythms, lush instrumentation, and a danceable vibe.",
  hiphop: "Hip-hop is a cultural movement and music style focused on rhythmic vocal delivery, DJing, and street culture.",
  jazz: "Jazz is an improvisational and expressive genre with deep roots in African-American culture and a strong emphasis on instrumentation.",
  metal: "Metal is a powerful genre featuring distorted guitars, aggressive vocals, and high energy, often exploring intense themes.",
  pop: "Pop music is widely appealing, with catchy melodies and production that aim for mainstream success.",
  reggae: "Reggae originates from Jamaica, featuring off-beat rhythms and messages of peace, love, and resistance.",
  rock: "Rock music is characterized by strong rhythms, electric guitars, and a rebellious attitude, evolving through many subgenres."
};

const genreImages: { [key: string]: string } = {
  blues: "/images/blues.jpg",
  classical: "/images/classical.jpg",
  country: "/images/country.jpg",
  disco: "/images/disco.jpg",
  hiphop: "/images/hiphop.jpg",
  jazz: "/images/jazz.jpg",
  metal: "/images/metal.jpg",
  pop: "/images/pop.jpg",
  reggae: "/images/reggae.jpg",
  rock: "/images/rock.jpg"
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [genre, setGenre] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setGenre(response.data.genre || 'Unknown');
    } catch (error) {
      console.error('Prediction error:', error);
      setGenre('Prediction failed.');
    }
    setLoading(false);
  };

  const genreKey = genre?.toLowerCase();
  const description = genreKey ? genreDescriptions[genreKey] : null;
  const imageUrl = genreKey ? genreImages[genreKey] : null;

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-100 to-white flex flex-col items-center justify-center p-6">
      <div className="bg-white shadow-2xl rounded-2xl p-8 w-full max-w-2xl text-center">
        <h1 className="text-4xl font-bold text-blue-800 mb-6">🎵 Music Genre Classifier</h1>
        <input
          type="file"
          accept="audio/wav, audio/mp3"
          onChange={handleFileChange}
          className="block w-full border border-gray-300 rounded-lg py-2 px-4 mb-4"
        />
        <button
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg shadow-md disabled:opacity-50 transition"
          onClick={handleUpload}
          disabled={loading || !file}
        >
          {loading ? '🔍 Predicting...' : '🎧 Upload & Predict'}
        </button>

        {genre && (
          <div className="mt-10 text-left">
            <h2 className="text-2xl font-bold text-green-700 mb-2">
              Predicted Genre: <span className="capitalize">{genre}</span>
            </h2>
            {description && (
              <p className="text-gray-700 text-lg leading-relaxed mb-4">
                {description}
              </p>
            )}
            {imageUrl && (
              <img
                src={imageUrl}
                alt={genre}
                className="rounded-lg shadow-lg mx-auto max-h-64 object-cover"
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

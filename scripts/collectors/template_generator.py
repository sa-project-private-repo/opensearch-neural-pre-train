"""Rule-based template generator for travel domain triplets."""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GeneratedTriplet:
    """Generated travel domain triplet."""

    query: str
    positive: str
    negative: str
    pair_type: str
    difficulty: str
    location: str
    metadata: Dict = field(default_factory=dict)


class TravelTemplateGenerator:
    """Generate travel domain triplets using rule-based templates.

    Creates query-positive-negative triplets for Korean travel/tourism
    domain by combining templates with location data.
    """

    # Korean administrative regions with aliases
    LOCATIONS = {
        "서울": {
            "full_name": "서울특별시",
            "aliases": ["서울시", "수도", "한양"],
            "attractions": [
                "경복궁",
                "창덕궁",
                "북촌한옥마을",
                "남산타워",
                "명동",
                "홍대",
                "강남",
                "인사동",
                "광화문",
                "동대문",
            ],
            "food": ["삼겹살", "비빔밥", "떡볶이", "치킨", "냉면"],
        },
        "부산": {
            "full_name": "부산광역시",
            "aliases": ["부산시"],
            "attractions": [
                "해운대",
                "광안리",
                "감천문화마을",
                "자갈치시장",
                "태종대",
                "용두산공원",
                "해동용궁사",
                "송정해수욕장",
            ],
            "food": ["회", "밀면", "돼지국밥", "씨앗호떡", "어묵"],
        },
        "제주": {
            "full_name": "제주특별자치도",
            "aliases": ["제주도", "제주시", "서귀포"],
            "attractions": [
                "한라산",
                "성산일출봉",
                "우도",
                "협재해수욕장",
                "천지연폭포",
                "만장굴",
                "용두암",
                "섭지코지",
            ],
            "food": ["흑돼지", "고등어회", "전복죽", "해산물", "귤"],
        },
        "경주": {
            "full_name": "경주시",
            "aliases": ["경북 경주"],
            "attractions": [
                "불국사",
                "석굴암",
                "첨성대",
                "동궁과월지",
                "대릉원",
                "경주월드",
                "보문단지",
                "황리단길",
            ],
            "food": ["황남빵", "찰보리빵", "한정식", "교동법주"],
        },
        "강원": {
            "full_name": "강원특별자치도",
            "aliases": ["강원도", "춘천", "강릉", "속초"],
            "attractions": [
                "설악산",
                "남이섬",
                "정동진",
                "경포대",
                "양양해변",
                "레고랜드",
                "오대산",
                "속초해수욕장",
            ],
            "food": ["막국수", "닭갈비", "순두부", "오징어순대", "감자옹심이"],
        },
        "인천": {
            "full_name": "인천광역시",
            "aliases": ["인천시"],
            "attractions": [
                "송도",
                "차이나타운",
                "월미도",
                "을왕리해수욕장",
                "소래포구",
                "강화도",
                "영종도",
            ],
            "food": ["짜장면", "자장면", "쫄면", "해물탕"],
        },
        "대구": {
            "full_name": "대구광역시",
            "aliases": ["대구시"],
            "attractions": [
                "팔공산",
                "앞산공원",
                "동성로",
                "서문시장",
                "김광석길",
                "수성못",
            ],
            "food": ["막창", "납작만두", "야끼우동", "뭉티기"],
        },
        "광주": {
            "full_name": "광주광역시",
            "aliases": ["광주시"],
            "attractions": [
                "무등산",
                "국립아시아문화전당",
                "양림동",
                "1913송정역시장",
            ],
            "food": ["떡갈비", "상추튀김", "오리탕", "육전"],
        },
        "전주": {
            "full_name": "전주시",
            "aliases": ["전북 전주"],
            "attractions": [
                "한옥마을",
                "전주향교",
                "덕진공원",
                "전주동물원",
                "오목대",
            ],
            "food": ["비빔밥", "콩나물국밥", "PNB초코파이", "한정식"],
        },
        "대전": {
            "full_name": "대전광역시",
            "aliases": ["대전시"],
            "attractions": [
                "대전엑스포과학공원",
                "계족산",
                "유성온천",
                "대청호",
            ],
            "food": ["성심당빵", "두부두루치기", "칼국수"],
        },
    }

    # Query templates
    QUERY_TEMPLATES = {
        "travel_recommend": [
            "{location} 여행 추천",
            "{location} 여행지 추천",
            "{location} 가볼만한 곳",
            "{location} 관광지 추천",
            "{location} 명소",
            "{location} 여행 코스",
            "{location} 여행 일정",
            "{location} 관광",
            "{location} 볼거리",
        ],
        "food_recommend": [
            "{location} 맛집 추천",
            "{location} 맛집",
            "{location} 음식 추천",
            "{location} 먹거리",
            "{location} 로컬 맛집",
            "{location} 유명 음식",
            "{location} 뭐 먹을까",
        ],
        "accommodation": [
            "{location} 숙소 추천",
            "{location} 호텔 추천",
            "{location} 숙박",
            "{location} 펜션",
            "{location} 어디서 자지",
        ],
        "activity": [
            "{location} 액티비티",
            "{location} 체험",
            "{location}에서 할 것",
            "{location} 즐길거리",
        ],
        "specific_attraction": [
            "{attraction} 가는 법",
            "{attraction} 입장료",
            "{attraction} 영업시간",
            "{attraction} 근처 맛집",
            "{attraction} 주변 볼거리",
        ],
    }

    # Response templates
    RESPONSE_TEMPLATES = {
        "travel_recommend": [
            "{location} 여행 추천 명소: {attractions}. {location}은(는) {description}으로 유명합니다.",
            "{location}에서 꼭 가봐야 할 곳은 {attractions}입니다. 특히 {highlight}이(가) 인기입니다.",
            "{location} 관광 코스 추천: 1일차 {day1}, 2일차 {day2}. {tip}",
        ],
        "food_recommend": [
            "{location} 대표 음식은 {foods}입니다. 특히 {highlight}이(가) 유명합니다.",
            "{location} 맛집 추천: {foods}를 드셔보세요. {location}만의 특색있는 맛을 느낄 수 있습니다.",
        ],
        "accommodation": [
            "{location} 숙소는 {area} 근처를 추천합니다. 교통이 편리하고 주변 관광지 접근성이 좋습니다.",
        ],
    }

    def __init__(
        self,
        output_dir: str = "data/v27.0/raw/generated",
        num_triplets: int = 10000,
        seed: int = 42,
    ):
        """Initialize generator.

        Args:
            output_dir: Directory to save generated triplets
            num_triplets: Number of triplets to generate
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_triplets = num_triplets
        self.seed = seed
        random.seed(seed)

    def _get_different_location(self, exclude: str) -> str:
        """Get a random location different from exclude."""
        locations = [loc for loc in self.LOCATIONS.keys() if loc != exclude]
        return random.choice(locations)

    def _generate_positive_text(
        self, location: str, pair_type: str
    ) -> str:
        """Generate positive document text."""
        loc_data = self.LOCATIONS[location]
        attractions = loc_data["attractions"]
        foods = loc_data.get("food", [])

        if pair_type == "travel_recommend":
            selected = random.sample(attractions, min(3, len(attractions)))
            return (
                f"{location} 여행 추천 명소: {', '.join(selected)}. "
                f"{location}은(는) 다양한 관광지와 문화체험으로 유명합니다. "
                f"특히 {selected[0]}이(가) 가장 인기있는 명소입니다."
            )

        elif pair_type == "food_recommend":
            if foods:
                selected = random.sample(foods, min(3, len(foods)))
                return (
                    f"{location} 대표 음식: {', '.join(selected)}. "
                    f"{location}을(를) 방문하면 꼭 {selected[0]}을(를) 드셔보세요."
                )
            else:
                return f"{location} 지역 특색 음식을 맛보실 수 있습니다."

        elif pair_type == "accommodation":
            area = attractions[0] if attractions else location
            return (
                f"{location} 숙소 추천: {area} 근처가 좋습니다. "
                f"교통이 편리하고 주요 관광지 접근이 용이합니다."
            )

        elif pair_type == "specific_attraction":
            attraction = random.choice(attractions)
            return (
                f"{attraction} 정보: {location}에 위치한 인기 관광지입니다. "
                f"대중교통으로 접근 가능하며 주변에 맛집이 많습니다."
            )

        else:
            selected = random.sample(attractions, min(2, len(attractions)))
            return f"{location} 여행: {', '.join(selected)} 등 다양한 볼거리가 있습니다."

    def _generate_negative_text(
        self, positive_location: str, pair_type: str
    ) -> Tuple[str, str]:
        """Generate hard negative document (similar format, different location).

        Returns:
            Tuple of (negative text, negative location)
        """
        neg_location = self._get_different_location(positive_location)
        neg_text = self._generate_positive_text(neg_location, pair_type)
        return neg_text, neg_location

    def generate(self) -> List[GeneratedTriplet]:
        """Generate travel domain triplets.

        Returns:
            List of GeneratedTriplet objects
        """
        triplets = []
        locations = list(self.LOCATIONS.keys())
        pair_types = list(self.QUERY_TEMPLATES.keys())

        for _ in range(self.num_triplets):
            # Select random location and pair type
            location = random.choice(locations)
            pair_type = random.choice(pair_types)

            # Generate query
            templates = self.QUERY_TEMPLATES[pair_type]
            template = random.choice(templates)

            if "{attraction}" in template:
                attraction = random.choice(
                    self.LOCATIONS[location]["attractions"]
                )
                query = template.format(attraction=attraction)
            else:
                query = template.format(location=location)

            # Generate positive
            positive = self._generate_positive_text(location, pair_type)

            # Generate hard negative (same type, different location)
            negative, neg_location = self._generate_negative_text(
                location, pair_type
            )

            # Determine difficulty based on location similarity
            if neg_location in ["서울", "인천"] and location in ["서울", "인천"]:
                difficulty = "hard"
            elif pair_type == "specific_attraction":
                difficulty = "medium"
            else:
                difficulty = "easy"

            triplets.append(
                GeneratedTriplet(
                    query=query,
                    positive=positive,
                    negative=negative,
                    pair_type=f"travel_{pair_type}",
                    difficulty=difficulty,
                    location=location,
                    metadata={
                        "source": "template_generator",
                        "negative_location": neg_location,
                    },
                )
            )

        logger.info(f"Generated {len(triplets):,} travel triplets")
        return triplets

    def save(self) -> Path:
        """Save generated triplets to JSONL file.

        Returns:
            Path to output file
        """
        output_file = self.output_dir / "travel_triplets.jsonl"
        triplets = self.generate()

        with open(output_file, "w", encoding="utf-8") as f:
            for triplet in triplets:
                data = {
                    "query": triplet.query,
                    "positive": triplet.positive,
                    "negative": triplet.negative,
                    "pair_type": triplet.pair_type,
                    "difficulty": triplet.difficulty,
                    "source": "template_generator",
                    "metadata": {
                        "location": triplet.location,
                        **triplet.metadata,
                    },
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(triplets):,} triplets to {output_file}")

        # Save statistics
        stats = self.get_stats(triplets)
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return output_file

    def get_stats(
        self, triplets: Optional[List[GeneratedTriplet]] = None
    ) -> Dict:
        """Get generation statistics."""
        if triplets is None:
            triplets = self.generate()

        stats = {
            "total": len(triplets),
            "by_location": {},
            "by_pair_type": {},
            "by_difficulty": {},
        }

        for triplet in triplets:
            loc = triplet.location
            pair_type = triplet.pair_type
            diff = triplet.difficulty

            stats["by_location"][loc] = stats["by_location"].get(loc, 0) + 1
            stats["by_pair_type"][pair_type] = (
                stats["by_pair_type"].get(pair_type, 0) + 1
            )
            stats["by_difficulty"][diff] = (
                stats["by_difficulty"].get(diff, 0) + 1
            )

        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generator = TravelTemplateGenerator()
    output_path = generator.save()
    print(f"Saved to: {output_path}")

"""Ejemplo de recomendacion de productos con vecinos mas cercanos."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


PRODUCTS = pd.DataFrame(
    [
        {"name": "Auriculares Nova", "category": "audio", "price": 79, "rating": 4.5, "popularity": 88},
        {"name": "Auriculares Pulse", "category": "audio", "price": 65, "rating": 4.2, "popularity": 76},
        {"name": "Altavoz Orbit", "category": "audio", "price": 115, "rating": 4.7, "popularity": 91},
        {"name": "Altavoz Mini", "category": "audio", "price": 45, "rating": 4.0, "popularity": 69},
        {"name": "Microfono Echo", "category": "audio", "price": 92, "rating": 4.6, "popularity": 83},
        {"name": "Auriculares Studio", "category": "audio", "price": 130, "rating": 4.8, "popularity": 95},
        {"name": "Teclado Mecanico", "category": "computacion", "price": 89, "rating": 4.6, "popularity": 90},
        {"name": "Mouse Inalambrico", "category": "computacion", "price": 39, "rating": 4.3, "popularity": 85},
        {"name": "Monitor Vista", "category": "computacion", "price": 210, "rating": 4.7, "popularity": 93},
        {"name": "Webcam Clara", "category": "computacion", "price": 59, "rating": 4.1, "popularity": 72},
        {"name": "Hub USB", "category": "computacion", "price": 28, "rating": 4.0, "popularity": 67},
        {"name": "Teclado Compacto", "category": "computacion", "price": 72, "rating": 4.5, "popularity": 81},
        {"name": "Botella Termica", "category": "hogar", "price": 25, "rating": 4.4, "popularity": 78},
        {"name": "Lampara de Escritorio", "category": "hogar", "price": 42, "rating": 4.2, "popularity": 71},
        {"name": "Taza Ceramica", "category": "hogar", "price": 18, "rating": 4.1, "popularity": 65},
        {"name": "Manta Suave", "category": "hogar", "price": 55, "rating": 4.7, "popularity": 86},
        {"name": "Organizador Modular", "category": "hogar", "price": 34, "rating": 4.3, "popularity": 74},
        {"name": "Cojin Ergonomico", "category": "hogar", "price": 48, "rating": 4.6, "popularity": 82},
    ]
)
FEATURE_COLUMNS = ["price", "rating", "popularity"]

train_products, test_products = train_test_split(
    PRODUCTS,
    test_size=0.25,
    random_state=42,
    stratify=PRODUCTS["category"],
)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_products[FEATURE_COLUMNS])
test_features = scaler.transform(test_products[FEATURE_COLUMNS])

model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_features, train_products["category"])
accuracy = accuracy_score(
    test_products["category"],
    model.predict(test_features),
)


def recommend(product_name: str) -> str:
    """Muestra y devuelve el producto mas parecido al producto indicado."""
    product_rows = PRODUCTS.loc[PRODUCTS["name"] == product_name]
    if product_rows.empty:
        available = ", ".join(PRODUCTS["name"])
        raise ValueError(
            f"Producto desconocido: {product_name!r}. Productos disponibles: {available}"
        )

    product = product_rows.iloc[[0]]
    product_features = scaler.transform(product[FEATURE_COLUMNS])
    distances, neighbor_positions = model.kneighbors(
        product_features,
        n_neighbors=len(train_products),
    )

    neighbors = train_products.iloc[neighbor_positions[0]].copy()
    neighbors["distance"] = distances[0]
    other_products = neighbors.loc[neighbors["name"] != product_name]
    same_category = other_products.loc[
        other_products["category"] == product.iloc[0]["category"]
    ]
    recommendation = (same_category if not same_category.empty else other_products).iloc[0]

    print(
        f"Producto recomendado para {product_name}: "
        f"{recommendation['name']} (categoria: {recommendation['category']})"
    )
    return str(recommendation["name"])


def main() -> None:
    print(f"Accuracy del modelo: {accuracy:.2f}")
    recommend("Auriculares Nova")


if __name__ == "__main__":
    main()
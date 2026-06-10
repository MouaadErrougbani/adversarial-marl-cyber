# Topologie réseau de CAGE4

ROUTERS = [
    "admin_network_subnet",
    "contractor_network_subnet",
    "internet_subnet",
    "office_network_subnet",
    "operational_zone_a_subnet",
    "operational_zone_b_subnet",
    "public_access_zone_subnet",
    "restricted_zone_a_subnet",
    "restricted_zone_b_subnet",
]

ROUTERS = [router + "_router" for router in ROUTERS]

ACCESSABLE_OFFLINE = {
    "admin_network_subnet_router": [
        "admin_network_subnet",
        "office_network_subnet",
        "public_access_zone_subnet",
    ],
    "contractor_network_subnet_router": [
        "contractor_network_subnet",
    ],
    "internet_subnet_router": [],
    "office_network_subnet_router": [
        "admin_network_subnet",
        "office_network_subnet",
        "public_access_zone_subnet",
    ],
    "operational_zone_a_subnet_router": [
        "operational_zone_a_subnet",
        "restricted_zone_a_subnet",
    ],
    "operational_zone_b_subnet_router": [
        "operational_zone_b_subnet",
        "restricted_zone_b_subnet",
    ],
    "public_access_zone_subnet_router": [
        "admin_network_subnet",
        "office_network_subnet",
        "public_access_zone_subnet",
    ],
    "restricted_zone_a_subnet_router": [
        "operational_zone_a_subnet",
        "restricted_zone_a_subnet",
    ],
    "restricted_zone_b_subnet_router": [
        "operational_zone_b_subnet",
        "restricted_zone_b_subnet",
    ],
}

MY_SUBNETS = {
    0: ["restricted_zone_a_subnet"],
    1: ["operational_zone_a_subnet"],
    2: ["restricted_zone_b_subnet"],
    3: ["operational_zone_b_subnet"],
    4: [
        "admin_network_subnet",
        "office_network_subnet",
        "public_access_zone_subnet",
    ],
}
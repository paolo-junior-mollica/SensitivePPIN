"""
api_clients — canonical home for all external API and dataset-download logic.

Each module in this package wraps a single external service or bulk-download
source.  Import each module explicitly; no names are re-exported from this
package-level __init__.

Modules
-------
uniprot          UniProt ID-mapping REST client
chembl           ChEMBL REST API client
opentargets      Open Targets GraphQL client (drug_disease_sources variant)
opentargets_chembl  Open Targets GraphQL client (drug_disease_sources_chembl variant)
ctgov            ClinicalTrials.gov REST client
reactome         Reactome ContentService REST client + cache
bulk_downloads   Generic file/repo/Zenodo download helpers
"""

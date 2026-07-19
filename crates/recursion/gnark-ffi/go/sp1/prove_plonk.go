package sp1

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
)

var plonkCacheMutex sync.Mutex
var cachedPlonkDataDir string
var cachedPlonkSCS constraint.ConstraintSystem
var cachedPlonkPK plonk.ProvingKey
var cachedPlonkVK plonk.VerifyingKey

func ProvePlonk(dataDir string, witnessPath string) Proof {
	// Sanity check the required arguments have been provided.
	if dataDir == "" {
		panic("dataDirStr is required")
	}
	os.Setenv("CONSTRAINTS_JSON", dataDir+"/"+constraintsJsonFile)

	if os.Getenv("SP1_PLONK_CACHE") == "1" {
		plonkCacheMutex.Lock()
		defer plonkCacheMutex.Unlock()
		if cachedPlonkDataDir == "" {
			cachedPlonkSCS, cachedPlonkPK, cachedPlonkVK = loadPlonkArtifacts(dataDir)
			cachedPlonkDataDir = dataDir
		} else if cachedPlonkDataDir != dataDir {
			panic(fmt.Sprintf(
				"SP1_PLONK_CACHE cannot switch circuit directories: cached %s, requested %s",
				cachedPlonkDataDir,
				dataDir,
			))
		} else {
			fmt.Printf("Using cached Plonk circuit and keys\n")
		}
		return provePlonkWithArtifacts(
			cachedPlonkSCS,
			cachedPlonkPK,
			cachedPlonkVK,
			witnessPath,
		)
	}

	scs, pk, vk := loadPlonkArtifacts(dataDir)
	return provePlonkWithArtifacts(scs, pk, vk, witnessPath)
}

func loadPlonkArtifacts(
	dataDir string,
) (constraint.ConstraintSystem, plonk.ProvingKey, plonk.VerifyingKey) {
	// Read the R1CS.
	start := time.Now()
	scsFile, err := os.Open(dataDir + "/" + plonkCircuitPath)
	if err != nil {
		panic(err)
	}
	scs := plonk.NewCS(ecc.BN254)
	scs.ReadFrom(scsFile)
	scsFile.Close()
	fmt.Printf("Reading Plonk SCS took %s\n", time.Since(start))

	// Read the proving key.
	start = time.Now()
	pkFile, err := os.Open(dataDir + "/" + plonkPkPath)
	if err != nil {
		panic(err)
	}
	pk := plonk.NewProvingKey(ecc.BN254)
	bufReader := bufio.NewReaderSize(pkFile, 1024*1024)
	pk.UnsafeReadFrom(bufReader)
	pkFile.Close()
	fmt.Printf("Reading Plonk proving key took %s\n", time.Since(start))

	// Read the verifier key.
	start = time.Now()
	vkFile, err := os.Open(dataDir + "/" + plonkVkPath)
	if err != nil {
		panic(err)
	}
	vk := plonk.NewVerifyingKey(ecc.BN254)
	vk.ReadFrom(vkFile)
	vkFile.Close()
	fmt.Printf("Reading Plonk verifying key took %s\n", time.Since(start))

	return scs, pk, vk
}

func provePlonkWithArtifacts(
	scs constraint.ConstraintSystem,
	pk plonk.ProvingKey,
	vk plonk.VerifyingKey,
	witnessPath string,
) Proof {
	// Read the file.
	start := time.Now()
	data, err := os.ReadFile(witnessPath)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Reading Plonk witness file took %s\n", time.Since(start))

	// Deserialize the JSON data into a slice of Instruction structs
	start = time.Now()
	var witnessInput WitnessInput
	err = json.Unmarshal(data, &witnessInput)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Deserializing Plonk witness took %s\n", time.Since(start))

	// Generate the witness.
	start = time.Now()
	assignment := NewCircuit(witnessInput)
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	publicWitness, err := witness.Public()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Generating Plonk witness took %s\n", time.Since(start))

	// Generate the proof.
	start = time.Now()
	proof, err := plonk.Prove(scs, pk, witness)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Generating Plonk proof took %s\n", time.Since(start))

	// Verify proof.
	start = time.Now()
	err = plonk.Verify(proof, vk, publicWitness)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Verifying Plonk proof took %s\n", time.Since(start))

	return NewSP1PlonkBn254Proof(&proof, witnessInput)
}
